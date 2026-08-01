/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import {
  surfaceClassName,
  inkClassName,
  strokeClassName,
  categoryBgClassName,
  categoryClassName,
  type CategoryRole,
} from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { flattenRagged, recordRole, RAGGED_CELL } from "./raggedLayout.js";
import type { RaggedNodeData } from "./types.js";

export type RaggedNodeType = Node<RaggedNodeData, "ragged">;

function Cell({
  value,
  role,
  filled,
}: {
  value: number | string;
  role?: CategoryRole;
  filled: boolean;
}): React.JSX.Element {
  return (
    <span
      className={clsx(
        "inline-flex items-center justify-center border text-xs tabular-nums",
        strokeClassName("secondary"),
        filled && role !== undefined
          ? `${categoryBgClassName(role)} text-white`
          : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
      )}
      style={{ width: RAGGED_CELL.width, height: RAGGED_CELL.height }}
    >
      {value}
    </span>
  );
}

function RowLabel({ text, role }: { text: string; role?: CategoryRole }): React.JSX.Element {
  return (
    <span
      className={clsx(
        "shrink-0 text-xs font-semibold",
        role !== undefined ? categoryClassName(role) : inkClassName("tertiary"),
      )}
      style={{ width: RAGGED_CELL.labelWidth }}
    >
      {text}
    </span>
  );
}

function SectionLabel({ text }: { text: string }): React.JSX.Element {
  return (
    // Line box pinned so `SECTION_LABEL_H` is enforced rather than inferred from font metrics.
    <span className={`text-xs font-semibold leading-[20px] ${inkClassName("primary")}`}>{text}</span>
  );
}

/**
 * Variable-length per-record lists, and the flat arrays they pack into.
 *
 * The point is the indirection: `values` holds every element back to back, `record_indices` says
 * who owns each one, and `offsets` says where each record's run begins — with -1 for a record that
 * contributed nothing, which is a different state from an empty run. Once the data is shaped this
 * way a whole dataset is filtered with a boolean mask over one column, so no per-record loop is
 * needed to answer a per-record question.
 *
 * `highlight` tints one record's cells across both the ragged and flat views, which is how a
 * reader follows a single record through the indirection.
 */
export function RaggedNode({ data }: NodeProps<RaggedNodeType>): React.JSX.Element {
  const { lists, title, highlight } = data;
  const showFlat = data.showFlat ?? true;
  const { values, recordIndices, offsets } = flattenRagged(lists);
  const isLit = (record: number) => highlight === undefined || record === highlight;

  return (
    <div
      className={clsx(
        "rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)]",
        surfaceClassName(data.surfaceRole ?? "elevated"),
        data.className,
      )}
    >
      {title !== undefined && (
        <div className={`mb-1.5 text-sm font-semibold leading-[24px] tracking-tight ${inkClassName("primary")}`}>
          {title}
        </div>
      )}

      <div className="flex flex-col" style={{ gap: RAGGED_CELL.rowGap }}>
        <SectionLabel text={data.raggedLabel ?? "per-record lists (ragged)"} />
        {lists.map((list, record) => (
          <div key={`r-${record}`} className="flex items-center" style={{ gap: RAGGED_CELL.gap }}>
            <RowLabel text={`r${record}`} role={recordRole(record)} />
            {list.length === 0 ? (
              <span
                className={`inline-flex items-center text-xs italic ${inkClassName("quaternary")}`}
                // A record with nothing in it still occupies a full row, so an absent run does not
                // silently shorten the node below its declared height.
                style={{ height: RAGGED_CELL.height }}
              >
                empty
              </span>
            ) : (
              list.map((v, i) => (
                <Cell key={i} value={v} role={recordRole(record)} filled={isLit(record)} />
              ))
            )}
          </div>
        ))}
      </div>

      {showFlat && (
        <div className="mt-3.5 flex flex-col" style={{ gap: RAGGED_CELL.rowGap }}>
          <SectionLabel text={data.flatLabel ?? "flat arrays"} />
          <div className="flex items-center" style={{ gap: RAGGED_CELL.gap }}>
            <RowLabel text="values" />
            {values.map((v, i) => (
              <Cell
                key={i}
                value={v}
                role={recordRole(recordIndices[i]!)}
                filled={isLit(recordIndices[i]!)}
              />
            ))}
          </div>
          <div className="flex items-center" style={{ gap: RAGGED_CELL.gap }}>
            <RowLabel text="record_indices" />
            {recordIndices.map((r, i) => (
              <Cell key={i} value={r} role={recordRole(r)} filled={false} />
            ))}
          </div>
          <div className="flex items-center" style={{ gap: RAGGED_CELL.gap }}>
            <RowLabel text="offsets" />
            {offsets.map((o, record) => (
              <Cell
                key={record}
                value={o < 0 ? "−1" : o}
                role={recordRole(record)}
                filled={isLit(record) && o >= 0}
              />
            ))}
          </div>
        </div>
      )}
      <NodeAnchorHandles />
    </div>
  );
}
