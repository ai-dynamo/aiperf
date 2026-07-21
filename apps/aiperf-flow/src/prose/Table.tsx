/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import {
  strokeClassName,
  inkClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../theme/tokens.js";

export type TableColumnAlign = "start" | "center" | "end";

export type TableColumn = {
  key: string;
  label: ReactNode;
  align?: TableColumnAlign;
};

export type TableRowTone = "neutral" | "success" | "warning" | "danger";

export type TableRow = Record<string, ReactNode> & { tone?: TableRowTone };

export type TableProps = {
  columns: TableColumn[];
  rows: TableRow[];
  className?: string;
};

const alignClassName: Record<TableColumnAlign, string> = {
  start: "text-start",
  center: "text-center",
  end: "text-end",
};

const toneCategory: Record<Exclude<TableRowTone, "neutral">, CategoryRole> = {
  success: "green",
  warning: "yellow",
  danger: "red",
};

function toneClassName(tone: TableRowTone | undefined): string | undefined {
  if (tone === undefined || tone === "neutral") {
    return undefined;
  }
  return categoryBgTintClassName(toneCategory[tone]);
}

/**
 * Data table with per-column alignment and per-row tone tinting.
 *
 * Renders a semantic `<table>` with `<thead>`/`<tbody>`; row tone maps onto
 * existing `CategoryRole` values as a subtle background tint rather than a
 * new tone token axis (mirrors `Callout.tsx`).
 */
export function Table({ columns, rows, className }: TableProps): React.JSX.Element {
  return (
    <table className={clsx("w-full rounded-none border-collapse text-sm", className)}>
      <thead>
        <tr className={clsx("border-b", strokeClassName("primary"))}>
          {columns.map((column) => (
            <th
              key={column.key}
              scope="col"
              className={clsx(
                "px-3 py-2 font-semibold",
                inkClassName("primary"),
                alignClassName[column.align ?? "start"],
              )}
            >
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr
            key={rowIndex}
            className={clsx(
              "border-b",
              strokeClassName("secondary"),
              toneClassName(row.tone),
            )}
          >
            {columns.map((column) => (
              <td
                key={column.key}
                className={clsx(
                  "px-3 py-2",
                  inkClassName("secondary"),
                  alignClassName[column.align ?? "start"],
                )}
              >
                {row[column.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
