// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubeconfig discovery and credential resolution.

use std::path::{Path, PathBuf};

use base64::Engine;
use serde::Deserialize;
use url::Url;

use super::error::KubeError;

/// Caller-selected Kubernetes authentication inputs.
#[derive(Clone, Debug, Default)]
pub struct KubeAuthOptions {
    /// Explicit kubeconfig path, taking precedence over environment discovery.
    pub kubeconfig: Option<PathBuf>,
    /// Explicit context name, taking precedence over kubeconfig current-context.
    pub context: Option<String>,
    /// Allow a server without certificate verification. This is never ambient.
    pub insecure_skip_tls_verify: bool,
}

/// Fully resolved API endpoint and authentication material.
#[derive(Clone, Debug)]
pub struct KubeCredentials {
    /// API hostname without a scheme.
    pub host: String,
    /// API port.
    pub port: u16,
    /// TLS server name.
    pub server_name: String,
    /// Bearer token used by the API request.
    pub token: Option<String>,
    /// Optional client certificate PEM.
    pub client_certificate_pem: Option<Vec<u8>>,
    /// Optional client key PEM.
    pub client_key_pem: Option<Vec<u8>>,
    /// Cluster CA PEM.
    pub ca_pem: Option<Vec<u8>>,
    /// Explicit TLS verification escape hatch.
    pub insecure_skip_tls_verify: bool,
}

impl KubeAuthOptions {
    /// Resolve kubeconfig precedence: explicit path, `KUBECONFIG`, then home config.
    pub fn kubeconfig_path(&self) -> Result<PathBuf, KubeError> {
        Self::kubeconfig_path_from(
            self.kubeconfig.as_deref(),
            std::env::var_os("KUBECONFIG"),
            std::env::var_os("HOME").map(PathBuf::from),
        )
    }

    /// Apply kubeconfig precedence to supplied process inputs.
    pub fn kubeconfig_path_from(
        explicit: Option<&Path>,
        kubeconfig: Option<std::ffi::OsString>,
        home: Option<PathBuf>,
    ) -> Result<PathBuf, KubeError> {
        if let Some(path) = explicit { return Ok(path.to_path_buf()); }
        if let Some(paths) = kubeconfig {
            if let Some(path) = std::env::split_paths(&paths).next() { return Ok(path); }
        }
        home.map(|home| home.join(".kube/config")).ok_or_else(|| {
            KubeError::Authentication("HOME is unset and no kubeconfig was selected".to_string())
        })
    }

    /// Resolve all selected kubeconfig entries into API credentials.
    pub fn resolve(&self) -> Result<KubeCredentials, KubeError> {
        let paths = self.kubeconfig_paths()?;
        let mut config = KubeConfig::default();
        for path in &paths {
            let source = std::fs::read_to_string(path)?;
            let mut next: KubeConfig = serde_yaml::from_str(&source)
                .map_err(|error| KubeError::Decode(format!("{}: {error}", path.display())))?;
            next.set_source(path);
            config.merge(next);
        }
        config.resolve(self, &paths[0])
    }

    /// Return every selected kubeconfig entry in Kubernetes precedence order.
    pub fn kubeconfig_paths(&self) -> Result<Vec<PathBuf>, KubeError> {
        if let Some(path) = &self.kubeconfig { return Ok(vec![path.clone()]); }
        if let Some(paths) = std::env::var_os("KUBECONFIG") {
            let paths: Vec<_> = std::env::split_paths(&paths).filter(|path| !path.as_os_str().is_empty()).collect();
            if !paths.is_empty() { return Ok(paths); }
        }
        Ok(vec![self.kubeconfig_path()?])
    }
}

/// Load service-account credentials for the controller reporting path.
pub fn in_cluster_credentials(
    host: String,
    port: u16,
    token_path: &Path,
    ca_path: &Path,
) -> Result<KubeCredentials, KubeError> {
    let token = std::fs::read_to_string(token_path)?.trim().to_string();
    if token.is_empty() {
        return Err(KubeError::Authentication("service-account token is empty".to_string()));
    }
    Ok(KubeCredentials {
        server_name: host.clone(),
        host,
        port,
        token: Some(token),
        client_certificate_pem: None,
        client_key_pem: None,
        ca_pem: Some(std::fs::read(ca_path)?),
        insecure_skip_tls_verify: false,
    })
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct KubeConfig {
    current_context: Option<String>,
    #[serde(default)]
    clusters: Vec<NamedCluster>,
    #[serde(default)]
    contexts: Vec<NamedContext>,
    #[serde(default)]
    users: Vec<NamedUser>,
}

impl KubeConfig {
    fn set_source(&mut self, path: &Path) {
        let source = path.to_path_buf();
        for cluster in &mut self.clusters { cluster.source = Some(source.clone()); }
        for user in &mut self.users { user.source = Some(source.clone()); }
    }

    fn merge(&mut self, next: Self) {
        if self.current_context.is_none() { self.current_context = next.current_context; }
        for cluster in next.clusters { if !self.clusters.iter().any(|entry| entry.name == cluster.name) { self.clusters.push(cluster); } }
        for context in next.contexts { if !self.contexts.iter().any(|entry| entry.name == context.name) { self.contexts.push(context); } }
        for user in next.users { if !self.users.iter().any(|entry| entry.name == user.name) { self.users.push(user); } }
    }

    fn resolve(&self, options: &KubeAuthOptions, path: &Path) -> Result<KubeCredentials, KubeError> {
        let context_name = options.context.as_deref().or(self.current_context.as_deref()).ok_or_else(|| {
            KubeError::Authentication("kubeconfig has no selected context".to_string())
        })?;
        let context = self.contexts.iter().find(|entry| entry.name == context_name).ok_or_else(|| {
            KubeError::Authentication(format!("kubeconfig context {context_name} does not exist"))
        })?;
        let cluster = self.clusters.iter().find(|entry| entry.name == context.context.cluster).ok_or_else(|| {
            KubeError::Authentication(format!("kubeconfig cluster {} does not exist", context.context.cluster))
        })?;
        let user = self.users.iter().find(|entry| entry.name == context.context.user).ok_or_else(|| {
            KubeError::Authentication(format!("kubeconfig user {} does not exist", context.context.user))
        })?;
        let (host, port) = split_host_port(&cluster.cluster.server)?;
        if cluster.cluster.insecure_skip_tls_verify.unwrap_or(false) && !options.insecure_skip_tls_verify {
            return Err(KubeError::Authentication("kubeconfig requests insecure TLS but --insecure-skip-tls-verify was not supplied".to_string()));
        }
        let token = resolve_token(&user.user, user.source.as_deref().unwrap_or(path))?;
        let client_certificate_pem = resolve_pem(
            user.user.client_certificate_data.as_deref(),
            user.user.client_certificate.as_deref(),
            user.source.as_deref().unwrap_or(path),
        )?;
        let client_key_pem = resolve_pem(
            user.user.client_key_data.as_deref(),
            user.user.client_key.as_deref(),
            user.source.as_deref().unwrap_or(path),
        )?;
        if token.is_none() && (client_certificate_pem.is_none() || client_key_pem.is_none()) {
            return Err(KubeError::Authentication("kubeconfig user has no bearer token, exec credential, or client cert/key pair".to_string()));
        }
        let ca_pem = resolve_pem(
            cluster.cluster.certificate_authority_data.as_deref(),
            cluster.cluster.certificate_authority.as_deref(),
            cluster.source.as_deref().unwrap_or(path),
        )?;
        Ok(KubeCredentials {
            server_name: host.clone(),
            host,
            port,
            token,
            client_certificate_pem,
            client_key_pem,
            ca_pem,
            insecure_skip_tls_verify: options.insecure_skip_tls_verify,
        })
    }
}

fn split_host_port(endpoint: &str) -> Result<(String, u16), KubeError> {
    let url = Url::parse(endpoint)
        .map_err(|error| KubeError::Authentication(format!("invalid Kubernetes API server {endpoint}: {error}")))?;
    if url.scheme() != "https" || url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return Err(KubeError::Authentication(format!("invalid Kubernetes API server {endpoint}")));
    }
    let host = url.host_str().ok_or_else(|| KubeError::Authentication(format!("invalid Kubernetes API server {endpoint}")))?;
    Ok((host.to_string(), url.port().unwrap_or(443)))
}

fn resolve_token(user: &User, config_path: &Path) -> Result<Option<String>, KubeError> {
    if let Some(token) = &user.token { return Ok(Some(token.clone())); }
    if let Some(path) = &user.token_file {
        let candidate = Path::new(path);
        let path = if candidate.is_absolute() { candidate.to_path_buf() } else { config_path.parent().unwrap_or(Path::new(".")).join(candidate) };
        return Ok(Some(std::fs::read_to_string(path)?.trim().to_string()));
    }
    user.exec.as_ref().map(run_exec_credential).transpose()
}

fn run_exec_credential(exec: &ExecCredential) -> Result<String, KubeError> {
    let output = std::process::Command::new(&exec.command)
        .args(&exec.args)
        .output()
        .map_err(KubeError::Io)?;
    if !output.status.success() {
        return Err(KubeError::Authentication(format!("exec credential command {} failed", exec.command)));
    }
    let credential: ExecCredentialResponse = serde_json::from_slice(&output.stdout)
        .map_err(|error| KubeError::Decode(format!("exec credential response: {error}")))?;
    credential.status.token.ok_or_else(|| KubeError::Authentication("exec credential response omitted status.token".to_string()))
}

fn resolve_pem(data: Option<&str>, file: Option<&str>, config_path: &Path) -> Result<Option<Vec<u8>>, KubeError> {
    if let Some(data) = data {
        return base64::engine::general_purpose::STANDARD
            .decode(data)
            .map(Some)
            .map_err(|error| KubeError::Decode(format!("base64 credential data: {error}")));
    }
    file.map(|file| {
        let candidate = Path::new(file);
        let resolved = if candidate.is_absolute() { candidate.to_path_buf() } else { config_path.parent().unwrap_or(Path::new(".")).join(candidate) };
        std::fs::read(resolved).map_err(KubeError::Io)
    }).transpose()
}

#[derive(Deserialize)]
struct NamedCluster { name: String, cluster: Cluster, #[serde(skip)] source: Option<PathBuf> }
#[derive(Deserialize)]
struct NamedContext { name: String, context: Context }
#[derive(Deserialize)]
struct NamedUser { name: String, user: User, #[serde(skip)] source: Option<PathBuf> }
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Cluster { server: String, certificate_authority: Option<String>, certificate_authority_data: Option<String>, insecure_skip_tls_verify: Option<bool> }
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Context { cluster: String, user: String }
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct User { token: Option<String>, token_file: Option<String>, client_certificate: Option<String>, client_certificate_data: Option<String>, client_key: Option<String>, client_key_data: Option<String>, exec: Option<ExecCredential> }
#[derive(Deserialize)]
struct ExecCredential { command: String, #[serde(default)] args: Vec<String> }
#[derive(Deserialize)]
struct ExecCredentialResponse { status: ExecCredentialStatus }
#[derive(Deserialize)]
struct ExecCredentialStatus { token: Option<String> }
