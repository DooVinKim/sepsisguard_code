## ════════════════════════════════════════════════════════════
## SepsisGuard 전처리 v2 — 공용 설정
##   모든 스크립트가 맨 위에서 source("00_config.R") 로 읽음
##   ※ 경로·피처·밴드·필터 값은 여기서만 바꿀 것
## ════════════════════════════════════════════════════════════
suppressMessages({
  library(data.table); library(arrow); library(lubridate)
  library(duckdb); library(DBI); library(glue); library(jsonlite)
})

DATA_DIR <- "../data"
EMR_DIR  <- "../EMR"
OUT_DIR  <- "../data/v2"          # v2 산출물은 전부 여기
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

VERSION <- "v2"

## ── 바이탈 피처 정의 ─────────────────────────────────────────
## itemid 는 d_items.csv 에서 반드시 재확인할 것 (label 로 grep).
## 같은 short 이름이 여러 itemid 에 걸리면(예: Temp) 한 컬럼으로 합쳐짐.
## lo/hi : 생리적 타당 범위 — 밖이면 NA(→ -1 마스크)
## f_to_c: TRUE 면 화씨→섭씨 변환 후 범위 적용
FEATURES <- data.table(
  itemid = c(220045, 220179, 220210, 220277, 223761, 223762,
             220180, 220181),
  short  = c("HR",   "SBP",  "RR",   "SpO2", "Temp", "Temp",
             "DBP",  "MAP"),
  lo     = c(20,     40,     4,      50,     29.4,   29.4,
             20,     30),
  hi     = c(250,    250,    60,     100,    43.3,   43.3,
             200,    220),
  f_to_c = c(FALSE,  FALSE,  FALSE,  FALSE,  TRUE,   FALSE,
             FALSE,  FALSE)
)
## 팀이 정한 추가 피처는 위 표에 행을 추가. 예:
##   GCS 세 항목(220739 eye, 223900 verbal, 223901 motor)은 short 를
##   따로 두거나("GCS_E","GCS_V","GCS_M") 02c 에서 합산 파생을 만들 것.
##   동맥압(220050/220051/220052)을 쓰려면 비침습과 같은 short 로 합칠지 결정.

VITAL_COLS <- c(unique(FEATURES$short), "SI")   # SI 는 02c 에서 파생

## ── 라벨 밴드 / 코호트 필터 (02c 가 명령행 인자로 덮어쓸 수 있음) ──
BAND_LOWER    <- 0     # label=1 : BAND_LOWER <  h <= BAND_UPPER
BAND_UPPER    <- 4
MIN_HIST_ROWS <- 2     # 양성 stay: 가장 이른 양성 행 앞에 실측 행 최소 개수
                       #   기존 win4 설계 = 6 (label_pos >= 7)
MIN_NEG_ROWS  <- 3     # 음성 stay: 전체 행 최소 개수 (기존 = 9)

## ── 분할 고정 ────────────────────────────────────────────────
## 기존 win4 의 split 을 환자 단위로 물려받고, 새로 들어온 환자만
## subject_id 해시로 70/15/15 배정 → 모든 v2 파일이 같은 test 를 가짐
REF_SPLIT_PARQUET <- file.path(DATA_DIR, "processed_labeled_win4.parquet")
SPLIT_FILE        <- file.path(OUT_DIR, "split_ids.json")

## ── EMR 윈도우 ───────────────────────────────────────────────
WIN_LAB   <- 24
WIN_MICRO <- 24
WIN_RADIO <- 72
NEG_PER_POS <- 20
K_LAST_LAB  <- 4      # 05: lab 줄에 적을 최근 측정값 개수 (토큰 길이 보고 조정)

## ── 유틸 ─────────────────────────────────────────────────────
## subject_id → [0,1) 결정적 해시 (부분집합 추출·신규 환자 split 배정에 공용)
id_hash <- function(ids) {
  x <- as.numeric(ids)
  ((x * 2654435761) %% 2^32) / 2^32
}
hash_split <- function(ids) {
  h <- id_hash(ids)
  fcase(h < 0.70, "train", h < 0.85, "val", default = "test")
}
log_step <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), sprintf(...), "\n")
