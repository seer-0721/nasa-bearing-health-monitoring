#!/usr/bin/env bash
set -euo pipefail

base_url="https://engineering.case.edu/sites/default/files"
root="${1:-data/CWRU}"

download_group() {
  local group="$1"
  shift
  mkdir -p "${root}/${group}"
  for id in "$@"; do
    local output="${root}/${group}/${id}.mat"
    if [[ -s "${output}" ]] && python -c 'from scipy.io import whosmat; import sys; whosmat(sys.argv[1])' "${output}" >/dev/null 2>&1; then
      echo "skip ${group}/${id}.mat"
      continue
    fi
    echo "download ${group}/${id}.mat"
    curl -L -f -C - --retry 5 --retry-all-errors --retry-delay 2 -o "${output}" "${base_url}/${id}.mat"
    python -c 'from scipy.io import whosmat; import sys; whosmat(sys.argv[1])' "${output}" >/dev/null
  done
}

download_group normal \
  97 98 99 100

download_group 12k_drive_end \
  105 118 130 144 156 \
  106 119 131 145 158 \
  107 120 132 146 159 \
  108 121 133 147 160 \
  169 185 197 \
  170 186 198 \
  171 187 199 \
  172 188 200 \
  209 222 234 246 258 \
  210 223 235 247 259 \
  211 224 236 248 260 \
  212 225 237 249 261 \
  3001 3005 \
  3002 3006 \
  3003 3007 \
  3004 3008

download_group 48k_drive_end \
  109 122 135 148 161 \
  110 123 136 149 162 \
  111 124 137 150 163 \
  112 125 138 151 164 \
  174 189 201 \
  175 190 202 \
  176 191 203 \
  177 192 204 \
  213 226 238 250 262 \
  214 227 239 251 263 \
  215 228 240 252 264 \
  217 229 241 253 265

download_group 12k_fan_end \
  278 282 294 298 302 \
  279 283 295 299 305 \
  280 284 296 300 306 \
  281 285 297 301 307 \
  274 286 313 310 \
  275 287 309 \
  276 288 311 \
  277 289 312 \
  270 290 315 \
  271 291 316 \
  272 292 317 \
  273 293 318

echo "done"
