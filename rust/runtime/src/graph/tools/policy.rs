// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic guarded shell-command policy for recorded replay.

use bytes::Bytes;

use super::environment::TraceEnvironmentError;

/// One command policy decision with explicit agent-visible rejection semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandDisposition {
    /// Send the authored command to the sandbox.
    Execute,
    /// Return a synthetic terminal command result without opening the sandbox.
    Synthetic(ToolCommandResult),
}

/// Terminal result emitted by one sandbox command attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCommandResult {
    /// Combined bounded stdout and stderr bytes.
    pub output: Bytes,
    /// Process-style result code.
    pub exit_code: i32,
    /// Clock-derived command duration in nanoseconds.
    pub duration_ns: u64,
    /// Whether execution reached its deadline.
    pub is_timed_out: bool,
    /// Whether the combined output exceeded the sandbox capture bound.
    pub is_output_truncated: bool,
}

impl ToolCommandResult {
    /// Build an ordinary terminal command result.
    pub fn completed(exit_code: i32, output: Bytes) -> Self {
        Self {
            output,
            exit_code,
            duration_ns: 0,
            is_timed_out: false,
            is_output_truncated: false,
        }
    }

    /// Build a fast deterministic deadline result for a fake or sandbox.
    pub fn timed_out(output: Bytes) -> Self {
        Self {
            output,
            exit_code: 124,
            duration_ns: 0,
            is_timed_out: true,
            is_output_truncated: false,
        }
    }

    /// Build the standard rejected-installer observation.
    pub fn installer_rejected() -> Self {
        Self {
            output: Bytes::from_static(
                b"recorded-agent replay blocked an installer command to preserve the task environment",
            ),
            exit_code: 127,
            duration_ns: 0,
            is_timed_out: false,
            is_output_truncated: false,
        }
    }

    /// Build the standard rejected-detachment observation.
    pub fn detaching_rejected() -> Self {
        Self {
            output: Bytes::from_static(
                b"recorded-agent replay blocked a detaching command to preserve sandbox containment",
            ),
            exit_code: 127,
            duration_ns: 0,
            is_timed_out: false,
            is_output_truncated: false,
        }
    }
}

/// Evaluates an authored shell command before any sandbox work occurs.
pub trait ToolCommandPolicy {
    /// Return a stable execution or synthetic-observation disposition.
    fn evaluate(&self, command: &str) -> Result<CommandDisposition, TraceEnvironmentError>;
}

/// Stock policy that rejects package installation in any top-level command segment.
#[derive(Clone, Copy, Debug, Default)]
pub struct GuardedToolCommandPolicy;

impl ToolCommandPolicy for GuardedToolCommandPolicy {
    fn evaluate(&self, command: &str) -> Result<CommandDisposition, TraceEnvironmentError> {
        for segment in top_level_segments(command) {
            if segment_starts_detaching_command(segment) {
                return Ok(CommandDisposition::Synthetic(
                    ToolCommandResult::detaching_rejected(),
                ));
            }
            if segment_starts_installer(segment) {
                return Ok(CommandDisposition::Synthetic(
                    ToolCommandResult::installer_rejected(),
                ));
            }
        }
        Ok(CommandDisposition::Execute)
    }
}

/// Return whether a shell command starts a known process-detachment utility.
pub(crate) fn contains_detaching_command(command: &str) -> bool {
    top_level_segments(command)
        .into_iter()
        .any(segment_starts_detaching_command)
}

fn segment_starts_detaching_command(segment: &str) -> bool {
    let tokens = shell_tokens(segment.trim());
    starts_detaching_command(&tokens)
        || nested_commands(segment)
            .iter()
            .flat_map(|command| top_level_segments(command))
            .any(segment_starts_detaching_command)
        || shell_command_payloads(&tokens)
            .iter()
            .flat_map(|command| top_level_segments(command))
            .any(segment_starts_detaching_command)
}

fn starts_detaching_command(tokens: &[String]) -> bool {
    let tokens = executable_tokens(tokens);
    tokens.first().is_some_and(|command| {
        matches!(
            strip_shell_expansions(command).as_str(),
            "setsid" | "nohup" | "disown"
        )
    })
}

fn shell_command_payloads(tokens: &[String]) -> Vec<&str> {
    let tokens = executable_tokens(tokens);
    let Some(executable) = tokens.first() else {
        return Vec::new();
    };
    let executable = strip_shell_expansions(executable);
    let executable = executable.rsplit('/').next().unwrap_or(&executable);
    if !matches!(executable, "bash" | "sh" | "zsh") {
        return Vec::new();
    }
    let mut arguments = tokens[1..].iter();
    while let Some(argument) = arguments.next() {
        if argument == "--" {
            break;
        }
        if argument.starts_with('-') && argument.contains('c') {
            return arguments.next().map(String::as_str).into_iter().collect();
        }
    }
    Vec::new()
}

fn top_level_segments(command: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let mut is_comment = false;
    let bytes = command.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if is_comment {
            if matches!(byte, b'\n' | b'\r') {
                segments.push(&command[start..index]);
                start = index + 1;
                is_comment = false;
            }
        } else if escaped {
            escaped = false;
        } else if byte == b'\\' && quote != Some(b'\'') {
            escaped = true;
        } else if matches!(byte, b'\'' | b'\"') {
            quote = if quote == Some(byte) {
                None
            } else if quote.is_none() {
                Some(byte)
            } else {
                quote
            };
        } else if quote.is_none()
            && byte == b'#'
            && (index == start || bytes[index - 1].is_ascii_whitespace())
        {
            is_comment = true;
        } else if quote.is_none() && matches!(byte, b';' | b'\n' | b'\r') {
            segments.push(&command[start..index]);
            start = index + 1;
        } else if quote.is_none()
            && index + 1 < bytes.len()
            && matches!((byte, bytes[index + 1]), (b'&', b'&') | (b'|', b'|'))
        {
            segments.push(&command[start..index]);
            index += 1;
            start = index + 1;
        }
        index += 1;
    }
    if quote.is_some() || escaped {
        return command.split_whitespace().collect();
    }
    segments.push(&command[start..]);
    segments
}

fn shell_tokens(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quote = None;
    let mut escaped = false;
    let characters = segment.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut expansion_depth = 0usize;
    let mut is_backtick = false;
    while index < characters.len() {
        let character = characters[index];
        if escaped {
            token.push(character);
            escaped = false;
        } else if character == '\\' && quote != Some('\'') {
            escaped = true;
        } else if matches!(character, '\'' | '\"') {
            quote = if quote == Some(character) {
                None
            } else if quote.is_none() {
                Some(character)
            } else {
                token.push(character);
                quote
            };
        } else if quote != Some('\'') && character == '`' {
            token.push(character);
            is_backtick = !is_backtick;
        } else if quote != Some('\'') && character == '$' && characters.get(index + 1) == Some(&'(')
        {
            token.push('$');
            token.push('(');
            expansion_depth = expansion_depth.saturating_add(1);
            index += 1;
        } else if expansion_depth > 0 && character == '(' {
            token.push(character);
            expansion_depth = expansion_depth.saturating_add(1);
        } else if expansion_depth > 0 && character == ')' {
            token.push(character);
            expansion_depth -= 1;
        } else if character.is_whitespace()
            && quote.is_none()
            && expansion_depth == 0
            && !is_backtick
        {
            if !token.is_empty() {
                tokens.push(std::mem::take(&mut token));
            }
        } else {
            token.push(character);
        }
        index += 1;
    }
    if escaped {
        token.push('\\');
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    tokens
}

fn starts_installer(tokens: &[String]) -> bool {
    let tokens = executable_tokens(tokens);
    let Some(command) = tokens.first() else {
        return false;
    };
    let command = strip_shell_expansions(command);
    is_installer_command(&command, tokens[0].as_str())
        || matches!(tokens, [python, flag, module, ..]
        if matches!(strip_shell_expansions(python).as_str(), "python" | "python3")
            && flag == "-m" && module == "pip")
}

fn is_installer_command(command: &str, source_word: &str) -> bool {
    const INSTALLERS: &[&str] = &[
        "pip", "pip3", "conda", "mamba", "apt", "apt-get", "yum", "dnf", "apk",
    ];
    INSTALLERS.contains(&command)
        || has_shell_expansion(source_word)
            && INSTALLERS
                .iter()
                .any(|installer| is_subsequence(command, installer))
}

fn has_shell_expansion(word: &str) -> bool {
    word.contains("$(") || word.contains("${") || word.contains('$') || word.contains('`')
}

fn is_subsequence(candidate: &str, target: &str) -> bool {
    let mut target = target.chars();
    candidate
        .chars()
        .all(|character| target.any(|target| target == character))
}

fn executable_tokens(mut tokens: &[String]) -> &[String] {
    loop {
        while tokens.first().is_some_and(|token| is_assignment(token)) {
            tokens = &tokens[1..];
        }
        if matches!(
            tokens.first().map(String::as_str),
            Some("then" | "do" | "else" | "elif")
        ) {
            tokens = &tokens[1..];
            continue;
        }
        if matches!(tokens.first().map(String::as_str), Some("sudo" | "env")) {
            tokens = &tokens[1..];
            while tokens
                .first()
                .is_some_and(|token| is_assignment(token) || token.starts_with('-'))
            {
                tokens = &tokens[1..];
            }
            continue;
        }
        if matches!(tokens.first().map(String::as_str), Some("command")) {
            tokens = &tokens[1..];
            let mut is_query = false;
            while tokens.first().is_some_and(|token| token.starts_with('-')) {
                is_query |= tokens
                    .first()
                    .is_some_and(|token| token.contains('v') || token.contains('V'));
                tokens = &tokens[1..];
            }
            if is_query {
                return &[];
            }
            continue;
        }
        return tokens;
    }
}

fn segment_starts_installer(segment: &str) -> bool {
    starts_installer(&shell_tokens(segment.trim()))
        || nested_commands(segment)
            .iter()
            .flat_map(|command| top_level_segments(command))
            .any(segment_starts_installer)
}

fn nested_commands(segment: &str) -> Vec<String> {
    let characters = segment.chars().collect::<Vec<_>>();
    let mut commands = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if escaped {
            escaped = false;
        } else if character == '\\' && quote != Some('\'') {
            escaped = true;
        } else if matches!(character, '\'' | '\"') {
            quote = if quote == Some(character) {
                None
            } else if quote.is_none() {
                Some(character)
            } else {
                quote
            };
        } else if quote != Some('\'') && character == '$' && characters.get(index + 1) == Some(&'(')
        {
            let end = skip_balanced_expansion(&characters, index + 2, '(', ')');
            if end > index + 2 {
                commands.push(characters[index + 2..end - 1].iter().collect());
            }
            index = end.saturating_sub(1);
        } else if quote != Some('\'') && character == '`' {
            let start = index + 1;
            index += 1;
            while index < characters.len() && characters[index] != '`' {
                index += 1;
            }
            if index < characters.len() {
                commands.push(characters[start..index].iter().collect());
            }
        }
        index += 1;
    }
    commands
}

fn strip_shell_expansions(token: &str) -> String {
    let characters = token.chars().collect::<Vec<_>>();
    let mut stripped = String::with_capacity(token.len());
    let mut index = 0;
    while index < characters.len() {
        match characters[index] {
            '$' if characters.get(index + 1) == Some(&'(') => {
                index = skip_balanced_expansion(&characters, index + 2, '(', ')');
            }
            '$' if characters.get(index + 1) == Some(&'{') => {
                index = skip_balanced_expansion(&characters, index + 2, '{', '}');
            }
            '$' => {
                index += 1;
                while characters
                    .get(index)
                    .is_some_and(|character| character == &'_' || character.is_ascii_alphanumeric())
                {
                    index += 1;
                }
            }
            '`' => {
                index += 1;
                while index < characters.len() && characters[index] != '`' {
                    index += 1;
                }
                index += usize::from(index < characters.len());
            }
            character => {
                stripped.push(character);
                index += 1;
            }
        }
    }
    stripped
}

fn skip_balanced_expansion(
    characters: &[char],
    mut index: usize,
    opening: char,
    closing: char,
) -> usize {
    let mut depth = 1usize;
    while index < characters.len() && depth > 0 {
        match characters[index] {
            character if character == opening => depth += 1,
            character if character == closing => depth -= 1,
            _ => {}
        }
        index += 1;
    }
    index
}

fn is_assignment(token: &str) -> bool {
    token.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
    })
}
