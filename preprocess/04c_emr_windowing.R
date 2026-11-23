## ════════════════════════════════════════════════════════════
## 4c. EMR 소스별 윈도잉 (v3 — ModernBERT 입력용)
##   04 대비 변경점
##   (1) lab : 윈도우를 인자로 (기본 48h). 윈도우 안의 측정값 전부 + 상대시각(h) 보존
##             가용성 절단은 04 와 동일 (coalesce(storetime, charttime) <= T)
##   (2) micro 결과 (신규) : T 기준 72h 안에 채취된 검체·검사 단위로
##             결과 저장시각(storetime) <= T 이면 결과(균명, 내성 항생제, 코멘트) 사용
##             아직 결과가 안 나왔으면 status = 'pending' (채취 사실은 T 시점에 알려진 정보)
##       storetime 은 (검체, 검사) 당 1개 = 최종 결과 시각. 채취→결과 중앙값 62h
##   (3) micro 채취 횟수 (24h) 는 04 와 동일하게 유지 → 04 산출물 재사용
##   (4) radiology 는 변경 없음 → 04 산출물 재사용
##
##   실행: Rscript 04c_emr_windowing.R b0_4_relax [lab_window_h=48]
##   출력: emr_labs_windowed_<tag>_L<win>.parquet
##         emr_micro_results_<tag>_M72.parquet
##         coverage_report_<tag>_L<win>.csv
## ════════════════════════════════════════════════════════════
source("00_config.R")
args <- commandArgs(trailingOnly = TRUE)
TAG     <- if (length(args) >= 1) args[1] else "b0_4_relax"
WIN_LAB <- if (length(args) >= 2) as.integer(args[2]) else 48L
WIN_MICRO_RES <- 72L

TP    <- file.path(OUT_DIR, sprintf("emr_timepoints_sampled_%s.parquet", TAG))
LABP  <- file.path(DATA_DIR, "labevents.parquet")
MICRO <- file.path(DATA_DIR, "microbiologyevents.csv")
OUT_LAB  <- file.path(OUT_DIR, sprintf("emr_labs_windowed_%s_L%d.parquet", TAG, WIN_LAB))
OUT_MRES <- file.path(OUT_DIR, sprintf("emr_micro_results_%s_M%d.parquet", TAG, WIN_MICRO_RES))

con <- dbConnect(duckdb())
dbExecute(con, "PRAGMA memory_limit='10GB'")
dbExecute(con, "PRAGMA threads=4")
dbExecute(con, glue("
  CREATE TEMP TABLE tp AS
  SELECT subject_id, hadm_id, stay_id, \"T\"::TIMESTAMP AS t_point
  FROM read_parquet('{TP}')
"))
n_tp <- dbGetQuery(con, "SELECT COUNT(*) n FROM tp")$n

## ── lab itemid 매핑 (04 와 동일) ─────────────────────────────
dbExecute(con, "
  CREATE TEMP TABLE lab_map (itemid INTEGER, lab_name VARCHAR);
  INSERT INTO lab_map VALUES
    (50813,'Lactate'),(52442,'Lactate'),(53154,'Lactate'),
    (51301,'WBC'),(51755,'WBC'),(51756,'WBC'),
    (51265,'Platelet'),
    (50912,'Creatinine'),(52546,'Creatinine'),
    (50885,'Bilirubin'),
    (51006,'BUN'),(52647,'BUN'),
    (50882,'Bicarbonate'),
    (50889,'CRP'),
    (51144,'Bands'),
    (51256,'Neutrophils');
")

## ── A. lab (WIN_LAB, 전체 측정값) ────────────────────────────
log_step("A. lab 윈도잉 %dh", WIN_LAB)
dbExecute(con, glue("
  CREATE TEMP TABLE lab_cand AS
  SELECT tp.stay_id, tp.subject_id, tp.t_point, m.lab_name,
         l.charttime::TIMESTAMP AS charttime,
         coalesce(l.storetime, l.charttime)::TIMESTAMP AS avail_time,
         l.valuenum, l.valueuom, l.flag, l.ref_range_lower, l.ref_range_upper
  FROM tp
  JOIN read_parquet('{LABP}') l
    ON l.hadm_id = tp.hadm_id
   AND l.charttime::TIMESTAMP <= tp.t_point
   AND l.charttime::TIMESTAMP >= tp.t_point - INTERVAL {WIN_LAB} HOUR
   AND l.valuenum IS NOT NULL
  JOIN lab_map m ON m.itemid = l.itemid
"))
leak <- dbGetQuery(con, "
  SELECT COUNT(*) AS n_rows_charttime,
         SUM(CASE WHEN avail_time > t_point THEN 1 ELSE 0 END) AS n_rows_not_yet_available
  FROM lab_cand")
leak$pct_rows_leaked <- round(leak$n_rows_not_yet_available / leak$n_rows_charttime * 100, 2)

dbExecute(con, glue("
  COPY (
    SELECT
      stay_id, subject_id, t_point, lab_name,
      arg_max(valuenum, charttime)                           AS last_val,
      arg_max(valueuom, charttime)                           AS uom,
      min(valuenum)                                          AS min_val,
      max(valuenum)                                          AS max_val,
      arg_max(valuenum, charttime) - arg_min(valuenum, charttime) AS delta_val,
      count(*)                                               AS n_meas,
      bool_or(flag IS NOT NULL AND flag <> '')               AS any_abnormal,
      arg_max(ref_range_lower, charttime)                    AS ref_low,
      arg_max(ref_range_upper, charttime)                    AS ref_high,
      list(valuenum ORDER BY charttime)                      AS vals,
      list(date_diff('minute', charttime, t_point) / 60.0 ORDER BY charttime) AS hrs_ago,
      date_diff('minute', max(charttime), any_value(t_point)) / 60.0 AS hours_since_last
    FROM lab_cand
    WHERE avail_time <= t_point
    GROUP BY stay_id, subject_id, t_point, lab_name
  ) TO '{OUT_LAB}' (FORMAT PARQUET)
"))
log_step("저장 %s", basename(OUT_LAB))

## ── B. micro 결과 (72h 채취, storetime <= T 이면 결과) ───────
log_step("B. micro 결과 윈도잉 %dh", WIN_MICRO_RES)
dbExecute(con, glue("
  CREATE TEMP TABLE mb AS
  SELECT hadm_id, micro_specimen_id, spec_type_desc, test_name,
         coalesce(charttime, chartdate::TIMESTAMP)::TIMESTAMP AS collect_time,
         coalesce(storetime, storedate::TIMESTAMP)::TIMESTAMP AS result_time,
         org_name, ab_name, interpretation, comments
  FROM read_csv_auto('{MICRO}')
  WHERE hadm_id IS NOT NULL
"))
## (시점, 검체, 검사) 단위. 결과는 result_time <= T 일 때만 노출
dbExecute(con, glue("
  COPY (
    SELECT tp.stay_id, tp.subject_id, tp.t_point,
           mb.micro_specimen_id, any_value(mb.spec_type_desc) AS spec_type_desc, mb.test_name,
           min(mb.collect_time) AS collect_time,
           max(mb.result_time)  AS result_time,
           CASE WHEN max(mb.result_time) IS NOT NULL AND max(mb.result_time) <= tp.t_point
                THEN 'resulted' ELSE 'pending' END AS status,
           CASE WHEN max(mb.result_time) <= tp.t_point THEN
             string_agg(DISTINCT mb.org_name, '; ') FILTER (WHERE mb.org_name IS NOT NULL AND mb.org_name <> 'CANCELLED')
           END AS organisms,
           CASE WHEN max(mb.result_time) <= tp.t_point THEN
             string_agg(DISTINCT mb.ab_name, ', ') FILTER (WHERE mb.interpretation = 'R')
           END AS resistant,
           CASE WHEN max(mb.result_time) <= tp.t_point THEN
             any_value(mb.comments) FILTER (WHERE mb.org_name IS NULL AND mb.comments IS NOT NULL)
           END AS comment,
           bool_or(mb.org_name = 'CANCELLED') AS cancelled
    FROM tp
    JOIN mb
      ON mb.hadm_id = tp.hadm_id
     AND mb.collect_time <= tp.t_point
     AND mb.collect_time >= tp.t_point - INTERVAL {WIN_MICRO_RES} HOUR
    GROUP BY tp.stay_id, tp.subject_id, tp.t_point, mb.micro_specimen_id, mb.test_name
  ) TO '{OUT_MRES}' (FORMAT PARQUET)
"))
log_step("저장 %s", basename(OUT_MRES))

## ── 커버리지 보고 ────────────────────────────────────────────
cov <- dbGetQuery(con, glue("
  SELECT
    (SELECT COUNT(DISTINCT stay_id || '_' || t_point) FROM read_parquet('{OUT_LAB}')) AS with_lab,
    (SELECT avg(n_meas) FROM read_parquet('{OUT_LAB}'))                               AS mean_n_meas,
    (SELECT avg(CASE WHEN n_meas > 1 THEN 1 ELSE 0 END) FROM read_parquet('{OUT_LAB}')) AS frac_multi_meas,
    (SELECT max(n_meas) FROM read_parquet('{OUT_LAB}'))                               AS max_n_meas,
    (SELECT COUNT(DISTINCT stay_id || '_' || t_point) FROM read_parquet('{OUT_MRES}')) AS with_micro72,
    (SELECT COUNT(DISTINCT CASE WHEN status = 'resulted' THEN stay_id || '_' || t_point END)
       FROM read_parquet('{OUT_MRES}'))                                                AS with_micro_result,
    (SELECT COUNT(DISTINCT CASE WHEN organisms IS NOT NULL THEN stay_id || '_' || t_point END)
       FROM read_parquet('{OUT_MRES}'))                                                AS with_organism
"))
rep <- data.table(tag = TAG, win_lab = WIN_LAB, n_tp = n_tp, cov, leak)
print(t(rep))
fwrite(rep, file.path(OUT_DIR, sprintf("coverage_report_%s_L%d.csv", TAG, WIN_LAB)))

dbDisconnect(con, shutdown = TRUE)
log_step("완료")
