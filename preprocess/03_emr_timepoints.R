## ════════════════════════════════════════════════════════════
## 3. EMR 시점 그리드 (v2) — 기존 03 과 동일 로직, 입력/출력만 버전화
##   실행: Rscript 03_emr_timepoints.R b0_4_relax
##   출력: OUT_DIR/emr_timepoints_full_<tag>.parquet
##         OUT_DIR/emr_timepoints_sampled_<tag>.parquet
##         OUT_DIR/patients_anchor.parquet (subject_id → anchor_year_group; 1회)
## ════════════════════════════════════════════════════════════
source("00_config.R")
args <- commandArgs(trailingOnly = TRUE)
TAG <- if (length(args) >= 1) args[1] else "b0_4_relax"
set.seed(42)

vital <- setDT(read_parquet(file.path(OUT_DIR, sprintf("processed_labeled_%s.parquet", TAG))))
grid <- vital[, .(subject_id, stay_id, T = hour_bucket, hours_to_sepsis_bucket,
                  sepsis_time, label, split)]
log_step("시점 %d / stay %d / label=1 %d", nrow(grid), uniqueN(grid$stay_id), sum(grid$label))

icu <- fread(file.path(DATA_DIR, "icustays.csv"),
             select = c("subject_id", "hadm_id", "stay_id", "intime", "outtime"))
icu[, `:=`(intime = as.POSIXct(intime, tz = "UTC"), outtime = as.POSIXct(outtime, tz = "UTC"))]
grid <- merge(grid, icu[, .(stay_id, hadm_id, icu_intime = intime, icu_outtime = outtime)],
              by = "stay_id", all.x = TRUE)
stopifnot(sum(is.na(grid$hadm_id)) == 0)
setcolorder(grid, c("subject_id", "hadm_id", "stay_id", "T", "icu_intime", "icu_outtime",
                    "hours_to_sepsis_bucket", "sepsis_time", "label", "split"))

write_parquet(grid, file.path(OUT_DIR, sprintf("emr_timepoints_full_%s.parquet", TAG)))

pos <- grid[label == 1]
neg <- grid[label == 0][, {
  n_take <- min(.N, nrow(pos[split == .BY$split]) * NEG_PER_POS)
  .SD[sample(.N, n_take)]
}, by = split]
sampled <- rbind(pos, neg); setorder(sampled, stay_id, T)
print(sampled[, .(n = .N, pos = sum(label == 1), neg = sum(label == 0)), by = split])
write_parquet(sampled, file.path(OUT_DIR, sprintf("emr_timepoints_sampled_%s.parquet", TAG)))

## 체크: 그리드 양성 수 == 바이탈 양성 행 수
stopifnot(nrow(pos) == sum(vital$label))

## anchor_year_group (1회) — 백본 겹침 분석용
anchor_pq <- file.path(OUT_DIR, "patients_anchor.parquet")
if (!file.exists(anchor_pq)) {
  pat <- fread(file.path(DATA_DIR, "patients.csv"),
               select = c("subject_id", "gender", "anchor_age", "anchor_year_group"))
  write_parquet(pat, anchor_pq)
  log_step("patients_anchor.parquet 저장 (%d명)", nrow(pat))
}
log_step("완료: %s", TAG)
