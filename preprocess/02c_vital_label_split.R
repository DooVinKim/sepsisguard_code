## ════════════════════════════════════════════════════════════
## 2c. 바이탈 시간버킷 집계 + 밴드 라벨 + 코호트 필터 + 고정 split + 정규화 (v2)
##   기존 02 / 02b / sepsis_3_join_sweep.R 대체.
##
## 실행:
##   Rscript 02c_vital_label_split.R [band_lower] [band_upper] [min_hist] [min_neg] [tag]
##   예) 기존 설계 재현 : Rscript 02c_vital_label_split.R 2 4 6 9 b2_4_strict
##       새 코호트      : Rscript 02c_vital_label_split.R 0 4 2 3 b0_4_relax
##   인자 없이 돌리면 00_config.R 의 값 사용.
##
## 출력 (OUT_DIR):
##   processed_labeled_<tag>.parquet        전체(split 컬럼 포함)
##   train|val|test_<tag>.parquet
##   vital_norm_stats_<tag>.json            train 기준 mean/sd (추론·빌드에서 재사용)
##   cohort_report_<tag>.csv                필터로 빠진 stay 집계
##   split_ids.json                         (없을 때만 생성, 이후 모든 실행이 공유)
## ════════════════════════════════════════════════════════════
source("00_config.R")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 4) {
  BAND_LOWER <- as.numeric(args[1]); BAND_UPPER <- as.numeric(args[2])
  MIN_HIST_ROWS <- as.integer(args[3]); MIN_NEG_ROWS <- as.integer(args[4])
}
TAG <- if (length(args) >= 5) args[5] else
  sprintf("b%g_%g_h%d", BAND_LOWER, BAND_UPPER, MIN_HIST_ROWS)
log_step("설정: band (%g, %g] / min_hist %d / min_neg %d / tag %s",
         BAND_LOWER, BAND_UPPER, MIN_HIST_ROWS, MIN_NEG_ROWS, TAG)

## ── STEP 1. 로드 + JOIN ──────────────────────────────────────
chart <- setDT(read_parquet(file.path(OUT_DIR, "chartevents_vitals_v2.parquet")))
sepsis3 <- setDT(read_parquet(file.path(DATA_DIR, "sepsis3_times.parquet")))
sepsis3[, sepsis_time := as.POSIXct(sepsis_time, tz = "UTC")]
n_sepsis_stays_total <- uniqueN(sepsis3$stay_id)

chart <- merge(chart, sepsis3[, .(stay_id, sepsis_time)], by = "stay_id", all.x = TRUE)
chart[, charttime := as.POSIXct(charttime, tz = "UTC")]
chart[, hours_to_sepsis := as.numeric(difftime(sepsis_time, charttime, units = "hours"))]

## ── STEP 2. 발병 이후 측정 제거 (기존 02:112 과 동일) ────────
chart <- chart[is.na(hours_to_sepsis) | hours_to_sepsis >= 0]
stays_after_cut <- unique(chart[!is.na(sepsis_time), stay_id])
n_sepsis_no_rows <- n_sepsis_stays_total - length(intersect(stays_after_cut, sepsis3$stay_id))
log_step("발병 전 행이 0개라 사라진 패혈증 stay: %d / %d", n_sepsis_no_rows, n_sepsis_stays_total)

## ── STEP 3. 1시간 버킷 평균 → wide ───────────────────────────
chart[, hour_bucket := floor_date(charttime, unit = "hour")]
hourly <- chart[!is.na(valuenum), .(valuenum = mean(valuenum), sepsis_time = first(sepsis_time)),
                by = .(subject_id, stay_id, hour_bucket, short)]
wide <- dcast(hourly, subject_id + stay_id + hour_bucket + sepsis_time ~ short,
              value.var = "valuenum")
## 피처 컬럼이 전혀 없는 경우(해당 버킷에 어떤 피처도 없음)는 dcast 에서 안 생김 → 행 자체가 없음

## 누락된 피처 컬럼 보강 (어떤 피처가 전체에서 0행이면 컬럼이 안 생김)
for (f in setdiff(unique(FEATURES$short), names(wide))) wide[, (f) := NA_real_]

wide[, hours_to_sepsis_bucket := as.numeric(difftime(sepsis_time, hour_bucket, units = "hours"))]
wide[, label := as.integer(!is.na(hours_to_sepsis_bucket) &
                           hours_to_sepsis_bucket >  BAND_LOWER &
                           hours_to_sepsis_bucket <= BAND_UPPER)]

## ── STEP 4. 파생: Shock Index (원 단위에서) ──────────────────
wide[, SI := HR / SBP]; wide[!is.finite(SI), SI := NA]
## (추가 파생이 필요하면 여기: 예) GCS 합산
## if (all(c("GCS_E","GCS_V","GCS_M") %in% names(wide))) wide[, GCS := GCS_E + GCS_V + GCS_M]

setcolorder(wide, c("subject_id", "stay_id", "hour_bucket", VITAL_COLS,
                    "hours_to_sepsis_bucket", "sepsis_time", "label"))
log_step("wide: %d행 / stay %d / label=1 %d행", nrow(wide), uniqueN(wide$stay_id), sum(wide$label))

## ── STEP 5. 코호트 필터 + 보고 ───────────────────────────────
setorder(wide, stay_id, hour_bucket)
wide[, row_in_stay := seq_len(.N), by = stay_id]
pos_info <- wide[label == 1, .(label_pos = min(row_in_stay), n_pos_rows = .N), by = stay_id]
stay_len <- wide[, .(stay_len = max(row_in_stay), is_sepsis = !is.na(sepsis_time[1])), by = stay_id]
info <- merge(stay_len, pos_info, by = "stay_id", all.x = TRUE)
info[, hist_rows := label_pos - 1L]

## 보고용 집계 (필터 적용 전)
rep <- list(
  tag                       = TAG,
  n_sepsis_stays_sepsis3    = n_sepsis_stays_total,
  n_sepsis_no_rows_before_onset = n_sepsis_no_rows,
  n_sepsis_stays_in_wide    = info[is_sepsis == TRUE, .N],
  n_sepsis_zero_pos_rows    = info[is_sepsis == TRUE & is.na(label_pos), .N],   # 밴드 안에 행 없음
  n_sepsis_hist_lt_min      = info[!is.na(label_pos) & hist_rows < MIN_HIST_ROWS, .N],
  n_sepsis_kept             = info[!is.na(label_pos) & hist_rows >= MIN_HIST_ROWS, .N],
  n_neg_stays_in_wide       = info[is_sepsis == FALSE, .N],
  n_neg_len_lt_min          = info[is_sepsis == FALSE & stay_len < MIN_NEG_ROWS, .N],
  n_neg_kept                = info[is_sepsis == FALSE & stay_len >= MIN_NEG_ROWS, .N]
)
hist_dist <- info[!is.na(label_pos), .N, by = .(hist_rows = pmin(hist_rows, 12))][order(hist_rows)]
cat("\n양성 전 실측 행 수 분포 (12 이상은 12로 묶음):\n"); print(hist_dist)
cat("\n코호트 보고:\n"); print(as.data.table(rep))
fwrite(as.data.table(rep), file.path(OUT_DIR, sprintf("cohort_report_%s.csv", TAG)))
fwrite(hist_dist, file.path(OUT_DIR, sprintf("cohort_histrows_%s.csv", TAG)))

keep_stays <- info[(!is.na(label_pos) & hist_rows >= MIN_HIST_ROWS) |
                   (is.na(label_pos) & is_sepsis == FALSE & stay_len >= MIN_NEG_ROWS), stay_id]
## 주의: 패혈증 stay 인데 밴드 안에 행이 없는 경우(label_pos NA & is_sepsis TRUE)는
##       음성으로 쓰지 않고 제외 (발병 전 상태를 음성으로 가르치지 않기 위함 — 기존과 동일)
wide <- wide[stay_id %in% keep_stays]
wide[, row_in_stay := NULL]
log_step("필터 후: %d행 / stay %d / 환자 %d / label=1 %d",
         nrow(wide), uniqueN(wide$stay_id), uniqueN(wide$subject_id), sum(wide$label))

## ── STEP 6. 고정 split ───────────────────────────────────────
if (!file.exists(SPLIT_FILE)) {
  if (file.exists(REF_SPLIT_PARQUET)) {
    ref <- setDT(read_parquet(REF_SPLIT_PARQUET))[, .(subject_id, split)]
    ref <- unique(ref)
    stopifnot(ref[, .N, by = subject_id][, all(N == 1)])   # 한 환자 한 split
    log_step("기존 win4 split 상속: train %d / val %d / test %d 명",
             ref[split == "train", .N], ref[split == "val", .N], ref[split == "test", .N])
  } else {
    ref <- data.table(subject_id = integer(), split = character())
    log_step("⚠️ 참조 parquet 없음 → 전원 해시 배정")
  }
  write_json(split(ref$subject_id, ref$split), SPLIT_FILE, auto_unbox = FALSE)
}
sp <- fromJSON(SPLIT_FILE)
ref <- rbindlist(lapply(names(sp), function(k) data.table(subject_id = sp[[k]], split = k)))

wide <- merge(wide, ref, by = "subject_id", all.x = TRUE)
n_new <- wide[is.na(split), uniqueN(subject_id)]
wide[is.na(split), split := hash_split(subject_id)]
log_step("split 배정: 상속 %d명 / 신규(해시) %d명",
         wide[, uniqueN(subject_id)] - n_new, n_new)
## 신규 환자도 파일에 기록해 다음 실행에서 동일 배정 보장
if (n_new > 0) {
  all_ids <- unique(wide[, .(subject_id, split)])
  write_json(split(all_ids$subject_id, all_ids$split), SPLIT_FILE, auto_unbox = FALSE)
}
print(wide[, .(patients = uniqueN(subject_id), rows = .N, pos = sum(label)), by = split])

## ── STEP 7. Z-score (train 통계) → json 저장 → NA → -1 ────────
tr <- wide[split == "train"]
norm <- lapply(VITAL_COLS, function(col) list(
  mean = mean(tr[[col]], na.rm = TRUE), sd = sd(tr[[col]], na.rm = TRUE),
  missing_rate_train = mean(is.na(tr[[col]]))))
names(norm) <- VITAL_COLS
write_json(norm, file.path(OUT_DIR, sprintf("vital_norm_stats_%s.json", TAG)),
           auto_unbox = TRUE, digits = 8)
cat("\n정규화 기준(train):\n")
for (col in VITAL_COLS) cat(sprintf("  %-6s mean=%8.3f sd=%8.3f missing=%5.1f%%\n",
  col, norm[[col]]$mean, norm[[col]]$sd, norm[[col]]$missing_rate_train * 100))
hi_miss <- names(Filter(function(x) x$missing_rate_train > 0.9, norm))
if (length(hi_miss)) cat("⚠️ train 결측 90% 초과 피처:", paste(hi_miss, collapse = ", "), "\n")

for (col in VITAL_COLS) {
  wide[, (col) := (get(col) - norm[[col]]$mean) / norm[[col]]$sd]
  wide[is.na(get(col)), (col) := -1]
}

## ── STEP 8. 저장 ─────────────────────────────────────────────
setorder(wide, stay_id, hour_bucket)
out <- file.path(OUT_DIR, sprintf("processed_labeled_%s.parquet", TAG))
write_parquet(wide, out)
for (s in c("train", "val", "test"))
  write_parquet(wide[split == s], file.path(OUT_DIR, sprintf("%s_%s.parquet", s, TAG)))
log_step("저장: %s (%.1f MB)", basename(out), file.size(out) / 1e6)

## ── 체크 ─────────────────────────────────────────────────────
cat("\n체크:\n")
cat(sprintf("  양성 행 h 범위: (%.2f, %.2f]  (기대: (%g, %g])\n",
            wide[label == 1, min(hours_to_sepsis_bucket)], wide[label == 1, max(hours_to_sepsis_bucket)],
            BAND_LOWER, BAND_UPPER))
cat(sprintf("  stay당 양성 행 최대: %d\n", wide[label == 1, .N, by = stay_id][, max(N)]))
gap <- wide[, .(gap = as.numeric(diff(hour_bucket), units = "hours")), by = stay_id]
cat(sprintf("  행 간격 >1h 비율: %.1f%%  (윈도우를 '행'으로 쓸 때 참고)\n", mean(gap$gap > 1) * 100))
