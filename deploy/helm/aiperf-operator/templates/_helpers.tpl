{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "aiperf-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "aiperf-operator.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aiperf-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aiperf-operator.labels" -}}
helm.sh/chart: {{ include "aiperf-operator.chart" . }}
{{ include "aiperf-operator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aiperf-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aiperf-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Operator pod selector labels — adds `app.kubernetes.io/component: operator` so
Service, NetworkPolicy, and PodDisruptionBudget selectors match *only* the
operator pod and not `helm test` hook pods (which linger as Completed between
test runs thanks to `hook-delete-policy: before-hook-creation`, and would
otherwise receive Service traffic despite not listening on health/results ports).
*/}}
{{- define "aiperf-operator.operatorSelectorLabels" -}}
{{ include "aiperf-operator.selectorLabels" . }}
app.kubernetes.io/component: operator
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "aiperf-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "aiperf-operator.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Default container image for AIPerfJob benchmark pods. Falls back to
"<image.repository>:<image.tag|Chart.AppVersion>" when defaults.image is unset
so users who override image.tag automatically get matching benchmark images.
*/}}
{{- define "aiperf-operator.defaultJobImage" -}}
{{- if .Values.defaults.image }}
{{- .Values.defaults.image }}
{{- else }}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
{{- end }}

{{/*
Service account name for `helm test` hook pods. A dedicated SA (separate
from the operator SA) keeps the test hook surface minimal: read-only get on
the AIPerfJob CRD and get/list pods in the release namespace.
*/}}
{{- define "aiperf-operator.testServiceAccountName" -}}
{{- printf "%s-tests" (include "aiperf-operator.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}
