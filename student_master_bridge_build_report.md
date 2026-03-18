# student_master and bridge build report

- student_master rows: 2500
- account_master rows: 2498
- student_key_bridge rows: 35784

## notes

- direct id fields are exported as exact identity templates or filtered_exact matches
- account fields are first aggregated into account_master, then mapped only when the organization tuple is uniquely identifiable
- unresolved account rows are kept unresolved instead of being force-matched
- this first bridge export focuses on the high-priority exact-id tables plus online-learning accounts
- large account event tables can be added in a later optimized pass if needed

## output files

- student_master.csv
- account_master.csv
- student_key_bridge.csv
- student_key_bridge_summary.csv
