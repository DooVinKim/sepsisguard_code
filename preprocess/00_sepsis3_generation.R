## Set Repositories
setRepositories(ind = 1:7)

## Load library
library(tidyverse)
library(data.table)
library(lubridate)
library(janitor)
library(glue)
library(arrow)
library(duckdb)
library(DBI)

## 경로 설정
WORK_DIR <- ".."
DATA_DIR <- "../data"
SQL_DIR  <- "../mimic-code/mimic-iv/concepts_duckdb"
DB_PATH  <- "../mimic4.duckdb"

## ============================================================
## 공용 함수: DB 연결 + 뷰 등록
##   - 배치마다 독립 실행할 수 있도록 매번 재연결 후 뷰 재등록
##   - DuckDB 파일(.duckdb)에는 derived 테이블이 이미 쌓여 있음
##   - 뷰(raw CSV/parquet 포인터)는 세션마다 다시 등록 필요
## ============================================================
connect_and_register <- function(create_new = FALSE) {
  if (create_new && file.exists(DB_PATH)) file.remove(DB_PATH)

  con <- dbConnect(duckdb(), dbdir = DB_PATH, read_only = FALSE)

  dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mimiciv_hosp")
  dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mimiciv_icu")
  dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mimiciv_derived")

  register_view <- function(schema, table, path) {
    path_fwd <- gsub("\\\\", "/", path)
    ext      <- tolower(tools::file_ext(path))
    reader   <- if (ext == "parquet") {
      sprintf("read_parquet('%s')", path_fwd)
    } else {
      sprintf("read_csv_auto('%s', header=true, ignore_errors=true)", path_fwd)
    }
    dbExecute(con, sprintf(
      "CREATE OR REPLACE VIEW %s.%s AS SELECT * FROM %s",
      schema, table, reader
    ))
  }

  register_view("mimiciv_hosp", "labevents",
    file.path(DATA_DIR, "labevents.parquet"))
  register_view("mimiciv_hosp", "prescriptions",
    file.path(DATA_DIR, "prescriptions.csv"))
  register_view("mimiciv_hosp", "microbiologyevents",
    file.path(DATA_DIR, "microbiologyevents.csv"))
  register_view("mimiciv_icu", "chartevents",
    file.path(DATA_DIR, "chartevents.parquet"))
  register_view("mimiciv_icu", "inputevents",
    file.path(DATA_DIR, "inputevents.parquet"))
  register_view("mimiciv_icu", "icustays",
    file.path(DATA_DIR, "icustays.csv"))
  register_view("mimiciv_icu", "outputevents",
    file.path(DATA_DIR, "outputevents.csv"))
  register_view("mimiciv_icu", "d_items",
    file.path(DATA_DIR, "d_items.csv"))

  cat("DB 연결 및 뷰 등록 완료\n")
  return(con)
}

run_sql_file <- function(con, rel_path) {
  full_path <- file.path(SQL_DIR, rel_path)
  sql <- paste(readLines(full_path, encoding = "UTF-8", warn = FALSE), collapse = "\n")
  start <- proc.time()
  tryCatch({
    dbExecute(con, sql)
    elapsed <- round((proc.time() - start)[3], 1)
    cat(sprintf("  [OK] %-50s  %.1f초\n", rel_path, elapsed))
  }, error = function(e) {
    cat(sprintf("  [ERROR] %s\n    → %s\n", rel_path, e$message))
    stop(sprintf("SQL 실행 실패: %s", rel_path))
  })
}

## ════════════════════════════════════════════════════════════
## ██ STEP 1. Parquet 변환 (최초 1회만 실행)
## ════════════════════════════════════════════════════════════
## chartevents(42GB), labevents(18GB), inputevents(3GB) 변환
## 변환 후 이 블록은 다시 실행할 필요 없음 (SKIP 메시지 출력됨)
## 소요 시간: 약 45분~1시간

tmp_con <- dbConnect(duckdb(), dbdir = ":memory:")

convert_to_parquet <- function(con, csv_path, parquet_path) {
  if (file.exists(parquet_path)) {
    cat(sprintf("  [SKIP] 이미 존재: %s\n", basename(parquet_path))); return()
  }
  cat(sprintf("  변환 중: %s\n", basename(csv_path)))
  start <- proc.time()
  dbExecute(con, sprintf(
    "COPY (SELECT * FROM read_csv_auto('%s', header=true)) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)",
    gsub("\\\\", "/", csv_path), gsub("\\\\", "/", parquet_path)
  ))
  cat(sprintf("  [OK] %.1f분\n", (proc.time() - start)[3] / 60))
}

convert_to_parquet(tmp_con,
  file.path(DATA_DIR, "chartevents.csv"),  file.path(DATA_DIR, "chartevents.parquet"))
convert_to_parquet(tmp_con,
  file.path(DATA_DIR, "labevents.csv"),    file.path(DATA_DIR, "labevents.parquet"))
convert_to_parquet(tmp_con,
  file.path(DATA_DIR, "inputevents.csv"),  file.path(DATA_DIR, "inputevents.parquet"))

dbDisconnect(tmp_con, shutdown = TRUE)
cat("=== STEP 1 완료 ===\n")


## ════════════════════════════════════════════════════════════
## ██ STEP 2. 배치 1 — 기반 테이블 + 소변량 (가벼움, 약 5~10분)
## ════════════════════════════════════════════════════════════
## icustay_times, icustay_hourly, weight_durations, urine_output

con <- connect_and_register(create_new = TRUE)   # ← 최초 실행: DB 새로 생성

for (f in c(
  "demographics/icustay_times.sql",
  "demographics/icustay_hourly.sql",
  "demographics/weight_durations.sql",
  "measurement/urine_output.sql"
)) run_sql_file(con, f)

dbDisconnect(con, shutdown = TRUE)
cat("=== STEP 2 (배치 1) 완료 ===\n")


## ════════════════════════════════════════════════════════════
## ██ STEP 3. 배치 2 — 측정값 (무거움, 약 30~60분)
## ════════════════════════════════════════════════════════════
## labevents 4회 스캔 (bg, enzyme, chemistry, cbc)
## chartevents 4회 스캔 (vitalsign, gcs, ventilator_setting, oxygen_delivery)

con <- connect_and_register(create_new = FALSE)  # ← 기존 DB에 이어서 추가

for (f in c(
  "measurement/bg.sql",
  "measurement/vitalsign.sql",
  "measurement/gcs.sql",
  "measurement/enzyme.sql",
  "measurement/chemistry.sql",
  "measurement/complete_blood_count.sql",
  "measurement/ventilator_setting.sql",
  "measurement/oxygen_delivery.sql",
  "measurement/urine_output_rate.sql"
)) run_sql_file(con, f)

dbDisconnect(con, shutdown = TRUE)
cat("=== STEP 3 (배치 2) 완료 ===\n")


## ════════════════════════════════════════════════════════════
## ██ STEP 4. 배치 3 — 약물 (약 10~20분)
## ════════════════════════════════════════════════════════════
## inputevents 4회 스캔 (혈압상승제)

con <- connect_and_register(create_new = FALSE)

for (f in c(
  "medication/antibiotic.sql",
  "medication/dobutamine.sql",
  "medication/dopamine.sql",
  "medication/epinephrine.sql",
  "medication/norepinephrine.sql"
)) run_sql_file(con, f)

dbDisconnect(con, shutdown = TRUE)
cat("=== STEP 4 (배치 3) 완료 ===\n")


## ════════════════════════════════════════════════════════════
## ██ STEP 5. 배치 4 — 최종 판정 (약 10~30분)
## ════════════════════════════════════════════════════════════
## ventilation → sofa → suspicion_of_infection → sepsis3

con <- connect_and_register(create_new = FALSE)

for (f in c(
  "treatment/ventilation.sql",
  "score/sofa.sql",
  "sepsis/suspicion_of_infection.sql",
  "sepsis/sepsis3.sql"
)) run_sql_file(con, f)

dbDisconnect(con, shutdown = TRUE)
cat("=== STEP 5 (배치 4) 완료 ===\n")


## ════════════════════════════════════════════════════════════
## ██ STEP 6. 결과 확인 및 저장
## ════════════════════════════════════════════════════════════

con <- connect_and_register(create_new = FALSE)

cat("\n=== sepsis3 결과 요약 ===\n")
print(dbGetQuery(con, "
  SELECT
    COUNT(*)                                          AS total_rows,
    COUNT(DISTINCT stay_id)                           AS unique_stays,
    SUM(CASE WHEN sepsis3 = TRUE  THEN 1 ELSE 0 END) AS sepsis_positive,
    SUM(CASE WHEN sepsis3 = FALSE THEN 1 ELSE 0 END) AS sepsis_negative
  FROM mimiciv_derived.sepsis3
"))

sepsis3_times <- dbGetQuery(con, "
  SELECT subject_id, stay_id,
         suspected_infection_time AS sepsis_time,
         sofa_time, sofa_score, sepsis3
  FROM mimiciv_derived.sepsis3
  WHERE sepsis3 = TRUE
")

cat(sprintf("\n패혈증 확진 ICU stay 수: %d\n", nrow(sepsis3_times)))
write_parquet(sepsis3_times, file.path(DATA_DIR, "sepsis3_times.parquet"))
cat("저장 완료: data/sepsis3_times.parquet\n")

dbDisconnect(con, shutdown = TRUE)
cat("\n=== 전체 완료 ===\n")
