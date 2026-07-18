// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export {
  attachSdkOrigin,
  stripSdkOrigin,
  stripSdkOriginsFromScene,
  type SdkOrigin,
} from "./provenance.js";
export {
  AIPERF_SDK_COMPONENTS,
  createSdkRegistry,
  GENERIC_SDK_COMPONENTS,
  lookupSdkComponent,
  type SdkRegistry,
} from "./registry.js";
export {
  SDK_ACTION_NAMES,
  type SceneFragment,
  type SdkActionName,
  type SdkComponentDefinition,
  type SdkComponentFactory,
  type SdkExpansionContext,
} from "./types.js";
