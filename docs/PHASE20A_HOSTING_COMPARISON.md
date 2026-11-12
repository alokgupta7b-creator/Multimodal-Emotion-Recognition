# Phase 20A — Hosting Provider & Infrastructure Comparison

**Date:** 2026-11-11  
**Target Workload:** Staging deployment for Multimodal Emotion Recognition (`DistilBERT` + `Wav2Vec2` + `ResNet18/MTCNN`, single active worker, `736.8 MB` runtime weights, ~1.5–1.8 GB active RSS, safe headroom `>= 4–8 GB` RAM)  
**Scope:** Staging only (zero purchases, zero external activations, no billing incurred)

---

## 1. Empirical Staging Resource Floor

| Metric | Measured Host Benchmark (Phase 19A / 19F / 20A) | Staging Requirement Minimum | Recommended Staging Allocation |
|---|---|---|---|
| **Active Runtime Weights** | `736,848,769` B (`702.7 MiB` across DistilBERT, Wav2Vec2, ResNet18 primary + fallback, legacy CNN) | `10 GB` persistent NVMe/disk (incl. OS + PyTorch wheels + Docker layers) | `50–100 GB` SSD/NVMe |
| **Backend RAM (3 Models Loaded)** | `1,521.2 MB` – `1,760.4 MB` RSS (`phase19a` / `phase18e`) | `4 GB` RAM (tight; single request) | **`8 GB` RAM** (`2–4 vCPU`) for OpenCV buffers & concurrency |
| **GPU VRAM (When CUDA Enabled)** | `787.7 MB` – `1,231.3 MB` peak allocated | Optional (`4+ GB` VRAM if GPU enabled) | CPU inference initially; GPU optional later |
| **Measured CPU vs GPU Latency** | **CPU (`12` threads):** load `9.325 s`, Text `204.5 ms`, Audio `671.3 ms`, Video `1026.4 ms` (~1.9 s combined)<br>**GPU (RTX 4050):** Video API p50 `550.6 ms`, 3-modality p50 `582.6 ms` | `< 60s` HTTP timeout | CPU sufficient for staging (`~1–2 s/request`) |
| **Transport Security** | `navigator.mediaDevices.getUserMedia()` blocked off `localhost` without HTTPS | Valid TLS certificate on staging domain | Caddy / Cloudflare / Managed HTTPS |

---

## 2. Hosting Provider & Plan Matrix

| Provider & Plan | vCPU / GPU | RAM / VRAM | Storage & Bandwidth | Python / Docker / Storage | HTTPS & Scaling | Observed Pricing (Ref.) | Staging Suitability |
|---|---|---|---|---|---|---|---|
| **1. Hostinger VPS — KVM 2** *(Recommended VPS)* | `2 vCPU` (AMD EPYC; throttles after CPU credits exhaust) / **No GPU** | `8 GB` RAM | `100 GB` NVMe SSD, `8 TB` bandwidth | Full root Linux, Docker & Docker Compose, local persistent NVMe volume | Reverse proxy HTTPS via Caddy / Let's Encrypt; vertical upgrade to KVM 4 | `~$8.99/mo` (24-mo upfront promo, `$14.99/mo` renewal); short-term monthly higher | **Best low-cost persistent staging VPS** (matches `8 GB` RAM requirement cleanly) |
| **Hostinger VPS — KVM 4** | `4 vCPU` / **No GPU** | `16 GB` RAM | `200 GB` NVMe SSD, `16 TB` bandwidth | Full root Linux, Docker & Docker Compose, local persistent NVMe | Reverse proxy HTTPS; comfortable concurrency headroom | `~$12.99/mo` (24-mo upfront promo, `$28.99/mo` renewal) | Best if multi-user CPU UAT runs concurrently |
| **2. AWS — EC2 `t3.large` (CPU)** | `2 vCPU` (burstable) / **No GPU** | `8 GiB` RAM | EBS (`$0.08/GB-mo`), `$0.09/GB` egress | Full Linux Docker control, S3 backup (`$0.023/GB-mo`) | ALB / Caddy TLS; horizontal ASG | `~$0.0832/hr` (`~$60.74/mo` 24/7 on-demand `us-east-1`, excluding EBS/egress) | Predictable on-demand hourly billing without multi-year lock-in |
| **AWS — EC2 `c7i.xlarge` (CPU)** | `4 vCPU` (4th-Gen Xeon, dedicated) / **No GPU** | `8 GiB` RAM | EBS + S3 | Full Linux Docker | High sustained CPU without burstable credit throttling | `~$0.1785/hr` (`~$130.31/mo` 24/7 `us-east-1`) | Production-grade CPU compute; overkill for initial staging |
| **AWS — EC2 `g4dn.xlarge` (GPU)** | `4 vCPU` + `1x NVIDIA T4` | `16 GiB` RAM / `16 GB` VRAM | NVMe + EBS | Native CUDA 12.4 Docker (`nvidia-container-toolkit`) | Managed TLS via ALB; scales via ASG | `~$0.526/hr` (`~$384/mo` 24/7 `us-east-1`) | Too expensive for continuous 24/7 staging |
| **3. Google Cloud — Cloud Run (CPU)** | Up to `4 vCPU` on-demand / **No GPU** | Up to `8–16 GiB` RAM | Ephemeral container + GCS FUSE / GCS bucket (`$0.02/GB-mo` + `5 GB` free US tier) | Native Docker container; **no persistent local disk** (needs GCS mount/bake); cold start loads ~703 MB weights | Automatic managed HTTPS; auto-scale; free tier `180k` vCPU-s & `360k` GiB-s/mo | Pay-per-use; `4 vCPU / 8 GiB` active only during requests (`~$0.000024/vCPU-s` + `~$0.0000025/GiB-s`) | Good intermittent staging option if cold-start (`~10s`+) accepted |
| **Google Cloud — Cloud Run (GPU)** | Min `4+ vCPU` + `1x NVIDIA L4` | `16+ GiB` RAM / `24 GB` VRAM | GCS / Artifact Registry | Managed CUDA container (no GPU free tier; billed whole instance lifecycle) | Automatic managed HTTPS | `~$0.000186/GPU-s` (`~$0.67/gpu-hr` + CPU/RAM; `~$480+/mo` if kept warm 24/7) | Use only on-demand / scaled-to-zero GPU testing |
| **4. Microsoft Azure — B4ms VM / `NC4as_T4_v3`** | `B4ms`: `4 vCPU` CPU-only<br>`NC4as_T4_v3`: `4 vCPU` + `1x T4` | `B4ms`: `16 GiB` RAM<br>`NC4as`: `28 GiB` RAM / `16 GB` VRAM | Managed Disk / Blob Storage | Full Linux Docker + CUDA on NC-series | Reverse proxy TLS or Azure App Gateway | `B4ms`: `~$0.166/hr` (`~$121–$125/mo` 24/7)<br>`NC4as_T4_v3`: `~$0.526/hr` (`~$380+/mo`, `~$0.15/hr` spot) | Enterprise Azure integration; `B4ms` is pricier than VPS for staging |
| **5. Serverless GPU Inference (RunPod / Modal)** | **Modal:** Per-sec `T4` / `L4`<br>**RunPod Flex:** Per-sec `RTX 4000 / A4000` (16GB) / `L4` (24GB) | `16 GB` – `24 GB` VRAM + configurable RAM | Persistent network volume (extra fee) | Custom Docker container with CUDA `12.4`; scale-to-zero | Managed HTTPS endpoints | **Modal T4:** `$0.000164/s` (`$0.59/hr` active) + CPU/RAM<br>**Modal L4:** `$0.80/hr`<br>**RunPod 16GB Flex:** `$0.58/hr` active | **Most cost-effective GPU option** if GPU acceleration is tested bursts-only (`<$5–$20/mo` light staging use) |

### Frontend Static Hosting Comparison (Separated Frontend Option)

| Provider | Tier Cost | Bandwidth | Commercial / Staging Policy | HTTPS & SPA Support |
|---|---|---|---|---|
| **Cloudflare Pages (Free)** *(Recommended Separated CDN)* | **`$0/mo`** | Unmetered / Unlimited | Allowed for commercial & staging (`500` builds/mo, unlimited branch previews) | Automatic global HTTPS + custom domains |
| **Azure Static Web Apps (Free / Standard)** | `$0/mo` (Free) / `$9/mo` (Standard) | Plan-tiered | Free is personal/hobby; Standard supports production SLA | Automatic managed SSL |
| **Netlify (Free)** | `$0/mo` (`300` credits pool) | `~15 GB/mo` credit-equivalent | Commercial projects allowed within credit pool | Unlimited deploy previews + HTTPS |
| **Vercel (Hobby)** | `$0/mo` | `100 GB/mo` | **Non-commercial / personal only** on Hobby tier | Branch previews + HTTPS |

---

## 3. CPU vs GPU Staging Assessment & Selection Rationale

1. **Do not purchase a 24/7 GPU VM (`$380–$550/mo`) for initial staging:** Direct measurement on CPU (`EMOTION_API_DEVICE=cpu`, `task-4155`) confirmed **all three models load in `9.325s`** and execute in **`204.5 ms` (Text), `671.3 ms` (Audio), and `1,026.4 ms` (Video)** — comfortably within the `60s` frontend request timeout without CUDA hardware.
2. **CPU vs GPU Recommendation for Staging:**
   - **Recommended Primary Staging Architecture (Cost-Optimal):**
     - **Frontend:** Cloudflare Pages Free (`$0/mo`, automatic HTTPS, unmetered bandwidth) **OR** bundled Nginx container behind Caddy on the same VPS when keeping zero external dependencies during early staging.
     - **Backend + Persistent Model Storage:** **Hostinger KVM 2 (`2 vCPU`, `8 GB RAM`, `100 GB NVMe`, `~$8.99/mo` promo / `$14.99/mo` renewal)** (or **AWS EC2 `t3.large`** / **DigitalOcean `s-4vcpu-8gb`** at `$48/mo` if strictly hourly month-to-month cancellation without multi-year commitment is preferred).
   - **Optional Burst GPU Path (Zero Rewrite):** Deploy the identical `Dockerfile.backend` with `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124` and `EMOTION_API_DEVICE=auto` on **Modal T4 (`$0.000164/s`)** or **RunPod Flex 16GB (`$0.58/hr`)** only during high-concurrency GPU benchmarking.
3. **Important Burstable CPU Caveat:** Both Hostinger KVM and AWS `t3` use CPU credit bursting (`throttles` if sustained at 100% CPU for long periods). Because staging traffic is low/intermittent, CPU credits replenish rapidly; before any public/production load, migrate to dedicated non-burstable compute (Hostinger KVM 4+ / AWS `c7i.xlarge` / GPU).

