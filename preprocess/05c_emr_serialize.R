## ════════════════════════════════════════════════════════════
## 5c. EMR 문서 직렬화 (v3 — ModernBERT 입력용)
##   05 대비 변경점
##   (1) lab 줄 : 윈도우(기본 48h) 안 측정값 전부 + 각 값의 상대시각 + 측정 횟수
##         "- Lactate (n=3): 1.1 [-40h] → 1.8 [-20h] → 3.2 [-2h] mmol/L (HIGH, ref 0.5-2, rising)"
##       추세는 보이는 값 전체의 첫 값 → 마지막 값 (05 처럼 화면과 어긋나지 않음)
##   (2) micro : 24h 채취 줄(05 와 동일) + 72h 결과 줄 (04c 산출물)
##         "- Blood (Blood Culture, Routine), collected 30h ago: pending"
##         "- Urine (URINE CULTURE), collected 50h ago, resulted 4h ago: ESCHERICHIA COLI; resistant: AMPICILLIN"
##       결과 줄은 최근 채취순 최대 MAX_MICRO_LINES 개, 넘으면 "(+N more)"
##   (3) radiology : 05 와 동일 (04 산출물 재사용)
##
##   실행: Rscript 05c_emr_serialize.R b0_4_relax [lab_window_h=48]
##   출력: emr_documents_<tag>_v3_L<win>.parquet
## ════════════════════════════════════════════════════════════
source("00_config.R")
args <- commandArgs(trailingOnly = TRUE)
TAG     <- if (length(args) >= 1) args[1] else "b0_4_relax"
WIN_LAB <- if (length(args) >= 2) as.integer(args[2]) else 48L
WIN_MICRO_RES <- 72L
TREND_TOL <- 0.05
MAX_MICRO_LINES <- 8

spine <- setDT(read_parquet(file.path(OUT_DIR, sprintf("emr_timepoints_sampled_%s.parquet", TAG))))
labs  <- setDT(read_parquet(file.path(OUT_DIR, sprintf("emr_labs_windowed_%s_L%d.parquet", TAG, WIN_LAB))))
micro <- setDT(read_parquet(file.path(OUT_DIR, sprintf("emr_micro_windowed_%s.parquet", TAG))))
mres  <- setDT(read_parquet(file.path(OUT_DIR, sprintf("emr_micro_results_%s_M%d.parquet", TAG, WIN_MICRO_RES))))
radio <- setDT(read_parquet(file.path(OUT_DIR, sprintf("emr_radio_windowed_%s.parquet", TAG))))

spine[, tk := as.numeric(T)]
for (d in list(labs, micro, mres, radio)) d[, tk := as.numeric(t_point)]

fmt_num <- function(x) trimws(formatC(x, format = "g", digits = 3))
fmt_h   <- function(h) sprintf("-%dh", as.integer(round(h)))

## ── (A) labs ─────────────────────────────────────────────────
labs[, status := fifelse(!is.na(ref_low) & !is.na(ref_high),
  fcase(last_val > ref_high, "HIGH", last_val < ref_low, "LOW", default = "normal"),
  fifelse(any_abnormal, "abnormal", "normal"))]
labs[, tol := fifelse(!is.na(ref_low) & !is.na(ref_high) & ref_high > ref_low,
                      (ref_high - ref_low) * TREND_TOL, 0)]
labs[, trend := fifelse(n_meas > 1,
  fcase(delta_val >  tol, ", rising",
        delta_val < -tol, ", falling",
        default = ", stable"), "")]
labs[, refpart := fifelse(!is.na(ref_low) & !is.na(ref_high),
  sprintf(", ref %s-%s", fmt_num(ref_low), fmt_num(ref_high)), "")]
## 전체 측정값 + 상대시각
labs[, seqpart := mapply(function(v, h) {
  paste(sprintf("%s [%s]", fmt_num(as.numeric(v)), fmt_h(as.numeric(h))), collapse = " → ")
}, vals, hrs_ago)]
labs[, line := sprintf("- %s (n=%d): %s %s (%s%s%s)",
  lab_name, n_meas, seqpart,
  fifelse(is.na(uom) | uom == "", "", uom),
  status, refpart, trend)]
labs[, line := gsub("  +", " ", line)]

lab_order <- c("Lactate","WBC","Neutrophils","Bands","Platelet",
               "Creatinine","BUN","Bilirubin","Bicarbonate","CRP")
labs[, lab_name := factor(lab_name, levels = lab_order)]
setorder(labs, stay_id, tk, lab_name)
lab_block <- labs[, .(labs_block = paste(line, collapse = "\n")), by = .(stay_id, tk)]

## ── (B) micro: 24h 채취 (05 와 동일) ─────────────────────────
micro[, drawn_line := sprintf(
  "- Cultures drawn in last %dh: %d. Blood culture: %s. Specimens: %s.",
  WIN_MICRO, n_cultures, fifelse(blood_culture, "yes", "no"),
  fifelse(is.na(specimen_types) | specimen_types == "", "n/a", specimen_types))]
drawn <- micro[, .(stay_id, tk, drawn_line)]

## ── (B') micro: 72h 결과 ─────────────────────────────────────
clean_comment <- function(x) {
  x <- gsub("_{2,}", "[redacted]", x)
  x <- gsub("[ \t\r\n]+", " ", x)
  x <- trimws(x)
  fifelse(nchar(x) > 80, paste0(substr(x, 1, 80), "..."), x)
}
mres <- mres[cancelled == FALSE | is.na(cancelled) | !is.na(organisms)]
mres[, c_ago := as.numeric(difftime(t_point, collect_time, units = "hours"))]
mres[, r_ago := as.numeric(difftime(t_point, result_time, units = "hours"))]
mres[, result := fcase(
  status == "pending", "pending",
  !is.na(organisms), paste0(organisms, fifelse(is.na(resistant), "", paste0("; resistant: ", resistant))),
  !is.na(comment), clean_comment(comment),
  default = "resulted, no organism reported")]
mres[, rline := fifelse(status == "pending",
  sprintf("- %s (%s), collected %dh ago: pending",
          spec_type_desc, test_name, as.integer(round(c_ago))),
  sprintf("- %s (%s), collected %dh ago, resulted %dh ago: %s",
          spec_type_desc, test_name, as.integer(round(c_ago)), as.integer(round(r_ago)), result))]
setorder(mres, stay_id, tk, c_ago)
res_block <- mres[, {
  n <- .N; keep <- head(rline, MAX_MICRO_LINES)
  if (n > MAX_MICRO_LINES) keep <- c(keep, sprintf("(+%d more)", n - MAX_MICRO_LINES))
  .(res_block = paste(keep, collapse = "\n"))
}, by = .(stay_id, tk)]

## ── (C) radiology (05 와 동일) ───────────────────────────────
clean_text <- function(x) {
  x <- gsub("_{2,}", "[redacted]", x)
  x <- gsub("\\[redacted\\]( ?\\[redacted\\])+", "[redacted]", x)
  x <- gsub("[=_*#-]{3,}", " ", x)
  x <- gsub("[ \t]*\n[ \t]*", " ", x)
  x <- gsub("[ \t]{2,}", " ", x)
  trimws(x)
}
IMP_MAX <- 1000
extract_impression <- function(txt) {
  m <- regmatches(txt, regexpr("(?si)IMPRESSION:\\s*(.*)", txt, perl = TRUE))
  body <- if (length(m)) sub("(?si)^IMPRESSION:\\s*", "", m, perl = TRUE) else txt
  body <- clean_text(body)
  if (nchar(body) > IMP_MAX) paste0(substr(body, 1, IMP_MAX), " ...") else body
}
radio[, impression := vapply(text, extract_impression, character(1))]
radio[, hrs_ago := round(as.numeric(difftime(t_point, note_time, units = "hours")))]
radio_block <- radio[, .(stay_id, tk, radio_block = sprintf('- (%dh ago) "%s"', hrs_ago, impression))]

## ── (D) 조립 ─────────────────────────────────────────────────
doc <- Reduce(function(a, b) merge(a, b, by = c("stay_id", "tk"), all.x = TRUE),
              list(spine, lab_block, drawn, res_block, radio_block))
doc[is.na(labs_block),  labs_block  := sprintf("No relevant labs measured in last %dh.", WIN_LAB)]
doc[is.na(drawn_line),  drawn_line  := sprintf("No cultures drawn in last %dh.", WIN_MICRO)]
doc[is.na(res_block),   res_block   := sprintf("No cultures collected in last %dh.", WIN_MICRO_RES)]
doc[is.na(radio_block), radio_block := sprintf("No imaging in last %dh.", WIN_RADIO)]

doc[, emr_text := sprintf(
"Recent laboratory results (last %dh):
%s

Microbiology (cultures drawn, last %dh):
%s

Culture results (collected in last %dh, results available at this time):
%s

Recent imaging impression (last %dh):
%s",
  WIN_LAB, labs_block, WIN_MICRO, drawn_line, WIN_MICRO_RES, res_block, WIN_RADIO, radio_block)]
doc[, text_len := nchar(emr_text)]

out <- doc[, .(subject_id, stay_id, t_point = T, label, split, emr_text, text_len)]
out_path <- file.path(OUT_DIR, sprintf("emr_documents_%s_v3_L%d.parquet", TAG, WIN_LAB))
write_parquet(out, out_path)

cat(sprintf("=== 완료: %s (%d 문서) ===\n", basename(out_path), nrow(out)))
cat("문서 길이(글자):\n"); print(summary(out$text_len))
cat(sprintf("micro 결과 줄 있음: %.1f%% / 그 중 resulted 포함: %.1f%%\n",
            mean(!grepl("No cultures collected", out$emr_text)) * 100,
            mean(grepl("resulted [0-9]+h ago", out$emr_text)) * 100))
cat("\n예시(양성, 길이 중앙):\n"); cat(out[label == 1][order(text_len)][ceiling(.N / 2), emr_text], "\n")
