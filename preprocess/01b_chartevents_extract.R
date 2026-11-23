## ════════════════════════════════════════════════════════════
## 1b. chartevents 추출 (v2)
##   기존 초기_바이탈_전처리.R 대체.
##   - chartevents.parquet(01 에서 변환됨) 을 DuckDB 로 읽어
##     ICU 환자 × FEATURES itemid 만 추출
##   - 화씨→섭씨 변환을 저장 **전** 에 수행 (기존 코드는 저장 후 변환)
##   - 타당 범위 밖 → NA
##   출력: OUT_DIR/chartevents_vitals_v2.parquet (long)
##         OUT_DIR/chartevents_vitals_v2_summary.csv (피처별 행수·NA율)
## ════════════════════════════════════════════════════════════
source("00_config.R")

CHART_PQ <- file.path(DATA_DIR, "chartevents.parquet")
ICU_CSV  <- file.path(DATA_DIR, "icustays.csv")
OUT_PQ   <- file.path(OUT_DIR, "chartevents_vitals_v2.parquet")

stopifnot(file.exists(CHART_PQ))

con <- dbConnect(duckdb())
dbExecute(con, "PRAGMA memory_limit='12GB'")
dbExecute(con, "PRAGMA threads=6")

dbWriteTable(con, "feat", FEATURES, overwrite = TRUE)

log_step("chartevents 스캔 시작 (itemid %d개)", nrow(FEATURES))
t0 <- proc.time()

## stay_id 가 NULL 인 행은 ICU 밖 측정 → 제외
dbExecute(con, glue("
  COPY (
    SELECT
      c.subject_id, c.hadm_id, c.stay_id,
      c.charttime::TIMESTAMP AS charttime,
      c.itemid, f.short,
      CASE WHEN f.f_to_c THEN (c.valuenum - 32) * 5.0 / 9.0
           ELSE c.valuenum END AS valuenum_raw,
      f.lo, f.hi
    FROM read_parquet('{CHART_PQ}') c
    JOIN feat f ON f.itemid = c.itemid
    JOIN (SELECT DISTINCT stay_id FROM read_csv_auto('{ICU_CSV}')) i
      ON i.stay_id = c.stay_id
    WHERE c.valuenum IS NOT NULL
  ) TO '{OUT_PQ}' (FORMAT PARQUET, COMPRESSION ZSTD)
"))
log_step("스캔 완료 %.1f분", (proc.time() - t0)[3] / 60)

## 범위 적용 + 요약은 R 쪽에서 (파일 크기는 수백 MB 수준)
chart <- setDT(read_parquet(OUT_PQ))
chart[, valuenum := fifelse(valuenum_raw >= lo & valuenum_raw <= hi, valuenum_raw, NA_real_)]
summ <- chart[, .(n = .N, n_out_of_range = sum(is.na(valuenum)),
                  pct_out = round(mean(is.na(valuenum)) * 100, 2),
                  median = round(median(valuenum, na.rm = TRUE), 2),
                  n_stay = uniqueN(stay_id)), by = short]
print(summ)
fwrite(summ, file.path(OUT_DIR, "chartevents_vitals_v2_summary.csv"))

chart <- chart[, .(subject_id, hadm_id, stay_id, charttime, itemid, short, valuenum)]
write_parquet(chart, OUT_PQ)
log_step("저장: %s (%.1f MB, %d행, stay %d)",
         basename(OUT_PQ), file.size(OUT_PQ) / 1e6, nrow(chart), uniqueN(chart$stay_id))

## 체크: 새 피처 중 행 수 0 인 것
missing <- setdiff(unique(FEATURES$short), summ$short)
if (length(missing)) cat("⚠️ 행이 0개인 피처:", paste(missing, collapse = ", "),
                         "— itemid 를 d_items 에서 다시 확인\n")
## 체크: Temp 중앙값이 35~40 이면 섭씨 변환 정상
cat("Temp 중앙값(섭씨여야 함):", summ[short == "Temp", median], "\n")

dbDisconnect(con, shutdown = TRUE)
