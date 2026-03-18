$ErrorActionPreference = "Stop"

function Normalize-Id {
    param([object]$Value)

    if ($null -eq $Value) { return "" }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return "" }
    return $text.ToLowerInvariant()
}

function To-DoubleOrNull {
    param([object]$Value)

    if ($null -eq $Value) { return $null }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }

    $number = 0.0
    if ([double]::TryParse($text, [ref]$number)) {
        return $number
    }

    return $null
}

function Convert-ToDate {
    param([object]$Value)

    if ($null -eq $Value) { return $null }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }

    if ($text -match '^\d{8}$') {
        try { return [datetime]::ParseExact($text, 'yyyyMMdd', $null) } catch { return $null }
    }

    $asDouble = 0.0
    if ([double]::TryParse($text, [ref]$asDouble)) {
        try { return [datetime]::FromOADate($asDouble) } catch { return $null }
    }

    try { return [datetime]$text } catch { return $null }
}

function Get-SchoolYearSemesterFromDate {
    param([datetime]$Date)

    if ($null -eq $Date) { return $null }

    if ($Date.Month -ge 8) {
        $startYear = $Date.Year
        $semester = "1"
    }
    elseif ($Date.Month -eq 1) {
        $startYear = $Date.Year - 1
        $semester = "1"
    }
    else {
        $startYear = $Date.Year - 1
        $semester = "2"
    }

    return [pscustomobject]@{
        school_year = ("{0}-{1}" -f $startYear, ($startYear + 1))
        semester    = $semester
    }
}

function Get-TermInfoFromScorePlan {
    param([string]$PlanValue)

    if ([string]::IsNullOrWhiteSpace($PlanValue)) { return $null }
    if ($PlanValue -match '^(?<sy>\d{4}-\d{4})-(?<sem>\d)$') {
        return [pscustomobject]@{
            school_year = $Matches["sy"]
            semester    = $Matches["sem"]
        }
    }
    return $null
}

function Get-TermOrder {
    param([string]$SchoolYear, [string]$Semester)

    if ($SchoolYear -notmatch '^(?<start>\d{4})-\d{4}$') { return -1 }
    return ([int]$Matches["start"] * 10 + [int]$Semester)
}

function Get-ExcelConnection {
    param([string]$Path)

    $resolved = Resolve-Path -LiteralPath $Path
    $connString = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$resolved;Extended Properties='Excel 12.0 Xml;HDR=YES;IMEX=1';"
    return New-Object System.Data.OleDb.OleDbConnection($connString)
}

function Get-FirstSheetName {
    param([string]$Path)

    $conn = Get-ExcelConnection -Path $Path
    try {
        $conn.Open()
        $schema = $conn.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)
        $sheet = $schema | Where-Object { $_.TABLE_NAME -like '*$' } | Select-Object -First 1
        if ($null -eq $sheet) {
            throw "No worksheet found in $Path"
        }
        return [string]$sheet.TABLE_NAME
    }
    finally {
        $conn.Close()
    }
}

function Get-ExcelColumns {
    param([string]$Path)

    $sheetName = Get-FirstSheetName -Path $Path
    $conn = Get-ExcelConnection -Path $Path
    try {
        $conn.Open()
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = "SELECT * FROM [$sheetName] WHERE 1=0"
        $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($cmd)
        $table = New-Object System.Data.DataTable
        [void]$adapter.FillSchema($table, [System.Data.SchemaType]::Source)
        return @($table.Columns | ForEach-Object { $_.ColumnName })
    }
    finally {
        $conn.Close()
    }
}

function Invoke-ExcelReader {
    param(
        [string]$Path,
        [string]$Query,
        [scriptblock]$RowHandler
    )

    $sheetName = Get-FirstSheetName -Path $Path
    $conn = Get-ExcelConnection -Path $Path
    try {
        $conn.Open()
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = ($Query -replace '\{SHEET\}', $sheetName)
        $reader = $cmd.ExecuteReader()
        try {
            while ($reader.Read()) {
                & $RowHandler $reader
            }
        }
        finally {
            $reader.Close()
        }
    }
    finally {
        $conn.Close()
    }
}

function Headers-Contain {
    param([string[]]$Headers, [string[]]$Required)

    $set = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($header in $Headers) { [void]$set.Add($header) }
    foreach ($item in $Required) { if (-not $set.Contains($item)) { return $false } }
    return $true
}

function Resolve-DataFiles {
    $aliases = @{}
    foreach ($file in (Get-ChildItem -LiteralPath "." -Filter "*.xlsx" | Where-Object { $_.Name -notlike '~$*' })) {
        $headers = Get-ExcelColumns -Path $file.FullName
        if (Headers-Contain -Headers $headers -Required @("XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM")) { $aliases["student_basic"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH", "TCNF", "ZF")) { $aliases["physical_test"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("LOGIN_NAME", "XM", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME", "ROLEID", "BFB")) { $aliases["online_learning"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "JXBH", "KCH", "KKXN", "KKXQ")) { $aliases["course_select"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "ZXJXJHH", "KCH", "KXH", "KCCJ", "JDCJ")) { $aliases["student_score"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XSBH", "TJNY", "SWLJSC")) { $aliases["internet_stats"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "YDRQ", "YDLBDM", "YDYYDM", "SPRQ", "SFZX")) { $aliases["status_change"] = $file; continue }
    }

    foreach ($alias in @("student_basic", "physical_test", "online_learning", "course_select", "student_score", "internet_stats", "status_change")) {
        if (-not $aliases.ContainsKey($alias)) {
            throw "Missing required alias: $alias"
        }
    }

    return $aliases
}

function Get-OrCreateStudentProfile {
    param([hashtable]$Map, [string]$StudentId)

    if (-not $Map.ContainsKey($StudentId)) {
        $Map[$StudentId] = [ordered]@{
            student_id = $StudentId
            XB         = ""
            NJ         = ""
            XSM        = ""
            ZYM        = ""
            YX         = ""
            ZY         = ""
            BJ         = ""
        }
    }

    return $Map[$StudentId]
}

function Set-IfEmpty {
    param([object]$Record, [string]$Field, [object]$Value)
    $text = if ($null -eq $Value) { "" } else { ([string]$Value).Trim() }
    if ([string]::IsNullOrWhiteSpace($text)) { return }
    if ([string]::IsNullOrWhiteSpace([string]$Record[$Field])) { $Record[$Field] = $text }
}

function Get-OrCreateSemesterRecord {
    param(
        [hashtable]$Map,
        [hashtable]$StudentProfiles,
        [string]$StudentId,
        [string]$SchoolYear,
        [string]$Semester
    )

    $key = "$StudentId|$SchoolYear|$Semester"
    if (-not $Map.ContainsKey($key)) {
        $profile = if ($StudentProfiles.ContainsKey($StudentId)) { $StudentProfiles[$StudentId] } else { $null }
        $Map[$key] = [ordered]@{
            student_id                    = $StudentId
            school_year                   = $SchoolYear
            semester                      = $Semester
            term_order                    = Get-TermOrder -SchoolYear $SchoolYear -Semester $Semester
            gender                        = if ($null -ne $profile) { $profile.XB } else { "" }
            grade                         = if ($null -ne $profile) { $profile.NJ } else { "" }
            college                       = if ($null -ne $profile) { if ($profile.YX) { $profile.YX } else { $profile.XSM } } else { "" }
            major                         = if ($null -ne $profile) { if ($profile.ZY) { $profile.ZY } else { $profile.ZYM } } else { "" }
            class_name                    = if ($null -ne $profile) { $profile.BJ } else { "" }
            selected_course_count         = 0
            score_course_count            = 0
            score_numeric_count           = 0
            score_sum                     = 0.0
            score_sq_sum                  = 0.0
            fail_course_count             = 0
            gpa_count                     = 0
            gpa_sum                       = 0.0
            credit_sum                    = 0.0
            resit_exam_count              = 0
            internet_month_count          = 0
            internet_hours_sum            = 0.0
            internet_diff_sum             = 0.0
            online_learning_bfb_snapshot  = ""
            physical_test_score           = ""
            physical_test_bmi             = ""
            risk_event_current            = 0
            risk_event_type_codes         = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
            risk_event_reason_codes       = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        }
    }

    return $Map[$key]
}

function Join-HashSet {
    param([object]$Value)
    if ($null -eq $Value) { return "" }
    return (($Value | Sort-Object) -join "|")
}

$files = Resolve-DataFiles

Write-Host "Step 1/5: rebuild student_master"
$studentProfiles = @{}

Invoke-ExcelReader -Path $files.student_basic.FullName -Query "SELECT [XH],[XB],[XSM],[ZYM] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    if ([string]::IsNullOrWhiteSpace($studentId)) { return }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "XB" -Value $reader["XB"]
    Set-IfEmpty -Record $profile -Field "XSM" -Value $reader["XSM"]
    Set-IfEmpty -Record $profile -Field "ZYM" -Value $reader["ZYM"]
}

Invoke-ExcelReader -Path $files.physical_test.FullName -Query "SELECT [XH],[XB],[NJ],[YX],[ZY],[BJ] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    if ([string]::IsNullOrWhiteSpace($studentId)) { return }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "XB" -Value $reader["XB"]
    Set-IfEmpty -Record $profile -Field "NJ" -Value $reader["NJ"]
    Set-IfEmpty -Record $profile -Field "YX" -Value $reader["YX"]
    Set-IfEmpty -Record $profile -Field "ZY" -Value $reader["ZY"]
    Set-IfEmpty -Record $profile -Field "BJ" -Value $reader["BJ"]
}

$onlineSnapshotMap = @{}
Invoke-ExcelReader -Path $files.online_learning.FullName -Query "SELECT [LOGIN_NAME],[ROLEID],[DEPARTMENT_NAME],[MAJOR_NAME],[CLASS_NAME],[BFB] FROM [{SHEET}]" -RowHandler {
    param($reader)
    if (([string]$reader["ROLEID"]).Trim() -ne "3") { return }
    $studentId = Normalize-Id -Value $reader["LOGIN_NAME"]
    if ([string]::IsNullOrWhiteSpace($studentId)) { return }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "YX" -Value $reader["DEPARTMENT_NAME"]
    Set-IfEmpty -Record $profile -Field "ZY" -Value $reader["MAJOR_NAME"]
    Set-IfEmpty -Record $profile -Field "BJ" -Value $reader["CLASS_NAME"]
    $bfbValue = To-DoubleOrNull -Value $reader["BFB"]
    if ($null -ne $bfbValue -and -not $onlineSnapshotMap.ContainsKey($studentId)) {
        $onlineSnapshotMap[$studentId] = $bfbValue
    }
}

$studentMasterFixed = $studentProfiles.Keys | Sort-Object | ForEach-Object {
    $profile = $studentProfiles[$_]
    [pscustomobject]@{
        student_id = $profile.student_id
        XB         = $profile.XB
        NJ         = $profile.NJ
        XSM        = $profile.XSM
        ZYM        = $profile.ZYM
        YX         = $profile.YX
        ZY         = $profile.ZY
        BJ         = $profile.BJ
    }
}

New-Item -ItemType Directory -Force -Path ".\prepared\01_keys", ".\prepared\03_datasets" | Out-Null
$studentMasterFixed | Export-Csv -LiteralPath ".\prepared\01_keys\student_master.csv" -NoTypeInformation -Encoding utf8
$studentMasterFixed | Export-Csv -LiteralPath ".\student_master.csv" -NoTypeInformation -Encoding utf8

Write-Host "Step 2/5: build base rows and aggregate semester features"
$semesterMap = @{}

Invoke-ExcelReader -Path $files.course_select.FullName -Query "SELECT [XH],[KKXN],[KKXQ] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    $schoolYear = ([string]$reader["KKXN"]).Trim()
    $semester = ([string]$reader["KKXQ"]).Trim()
    if ([string]::IsNullOrWhiteSpace($studentId) -or [string]::IsNullOrWhiteSpace($schoolYear) -or [string]::IsNullOrWhiteSpace($semester)) { return }
    if (-not $studentProfiles.ContainsKey($studentId)) { return }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $schoolYear -Semester $semester
    $record.selected_course_count += 1
}

Invoke-ExcelReader -Path $files.student_score.FullName -Query "SELECT [XH],[ZXJXJHH],[XF],[CXBKBZ],[KCCJ],[JDCJ] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    $termInfo = Get-TermInfoFromScorePlan -PlanValue ([string]$reader["ZXJXJHH"])
    if ([string]::IsNullOrWhiteSpace($studentId) -or $null -eq $termInfo) { return }
    if (-not $studentProfiles.ContainsKey($studentId)) { return }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester

    $scoreValue = To-DoubleOrNull -Value $reader["KCCJ"]
    $isValidScore = ($null -ne $scoreValue -and $scoreValue -ge 0 -and $scoreValue -le 100)
    if ($isValidScore) {
        $record.score_course_count += 1
        $record.score_numeric_count += 1
        $record.score_sum += $scoreValue
        $record.score_sq_sum += ($scoreValue * $scoreValue)
        if ($scoreValue -lt 60) { $record.fail_course_count += 1 }
    }

    $gpaValue = To-DoubleOrNull -Value $reader["JDCJ"]
    if ($null -ne $gpaValue) {
        $record.gpa_count += 1
        $record.gpa_sum += $gpaValue
    }

    $creditValue = To-DoubleOrNull -Value $reader["XF"]
    if ($null -ne $creditValue -and $creditValue -gt 0) { $record.credit_sum += $creditValue }
    if ($isValidScore -and -not [string]::IsNullOrWhiteSpace([string]$reader["CXBKBZ"]) -and ([string]$reader["CXBKBZ"]).Trim() -ne "01") {
        $record.resit_exam_count += 1
    }
}

Invoke-ExcelReader -Path $files.internet_stats.FullName -Query "SELECT [XSBH],[TJNY],[SWLJSC],[XXPJZ] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XSBH"]
    if ([string]::IsNullOrWhiteSpace($studentId)) { return }
    if (-not $studentProfiles.ContainsKey($studentId)) { return }
    $termInfo = Get-SchoolYearSemesterFromDate -Date (Convert-ToDate -Value $reader["TJNY"])
    if ($null -eq $termInfo) { return }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester
    $record.internet_month_count += 1

    $hoursValue = To-DoubleOrNull -Value $reader["SWLJSC"]
    if ($null -ne $hoursValue) { $record.internet_hours_sum += $hoursValue }

    $schoolAvgValue = To-DoubleOrNull -Value $reader["XXPJZ"]
    if ($null -ne $schoolAvgValue -and $null -ne $hoursValue) {
        $record.internet_diff_sum += ($hoursValue - $schoolAvgValue)
    }
}

$physicalYearMap = @{}
Invoke-ExcelReader -Path $files.physical_test.FullName -Query "SELECT [XH],[TCNF],[ZF],[BMI] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    $yearText = ([string]$reader["TCNF"]).Trim()
    if ([string]::IsNullOrWhiteSpace($studentId) -or $yearText -notmatch '^\d{4}$') { return }
    if (-not $studentProfiles.ContainsKey($studentId)) { return }
    $schoolYear = ("{0}-{1}" -f $yearText, ([int]$yearText + 1))
    $key = "$studentId|$schoolYear"
    if (-not $physicalYearMap.ContainsKey($key)) {
        $physicalYearMap[$key] = [ordered]@{
            zf_sum   = 0.0
            zf_count = 0
            bmi_text = ""
        }
    }
    $zfValue = To-DoubleOrNull -Value $reader["ZF"]
    if ($null -ne $zfValue) {
        $physicalYearMap[$key].zf_sum += $zfValue
        $physicalYearMap[$key].zf_count += 1
    }
    if ([string]::IsNullOrWhiteSpace([string]$physicalYearMap[$key].bmi_text) -and -not [string]::IsNullOrWhiteSpace([string]$reader["BMI"])) {
        $physicalYearMap[$key].bmi_text = ([string]$reader["BMI"]).Trim()
    }
}

Write-Host "Step 3/5: derive risk events"
$highRiskTypeSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($riskType in @("1", "3", "11", "13")) { [void]$highRiskTypeSet.Add($riskType) }

Invoke-ExcelReader -Path $files.status_change.FullName -Query "SELECT [XH],[YDRQ],[YDLBDM],[YDYYDM],[SPRQ],[BY1],[SFZX] FROM [{SHEET}]" -RowHandler {
    param($reader)
    $studentId = Normalize-Id -Value $reader["XH"]
    if ([string]::IsNullOrWhiteSpace($studentId)) { return }
    if (-not $studentProfiles.ContainsKey($studentId)) { return }
    $typeCode = ([string]$reader["YDLBDM"]).Trim()
    $reasonCode = ([string]$reader["YDYYDM"]).Trim()
    $sfzx = ([string]$reader["SFZX"]).Trim()
    $isHighRisk = $highRiskTypeSet.Contains($typeCode) -or $sfzx -eq "0"
    if (-not $isHighRisk) { return }

    $eventDate = Convert-ToDate -Value $reader["YDRQ"]
    if ($null -eq $eventDate) { $eventDate = Convert-ToDate -Value $reader["SPRQ"] }
    if ($null -eq $eventDate) { $eventDate = Convert-ToDate -Value $reader["BY1"] }
    $termInfo = Get-SchoolYearSemesterFromDate -Date $eventDate
    if ($null -eq $termInfo) { return }

    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester
    $record.risk_event_current = 1
    if (-not [string]::IsNullOrWhiteSpace($typeCode)) { [void]$record.risk_event_type_codes.Add($typeCode) }
    if (-not [string]::IsNullOrWhiteSpace($reasonCode)) { [void]$record.risk_event_reason_codes.Add($reasonCode) }
}

Write-Host "Step 4/5: finalize rows and compute next-term label"
$rowsByStudent = @{}
foreach ($key in $semesterMap.Keys) {
    $record = $semesterMap[$key]
    if (-not $rowsByStudent.ContainsKey($record.student_id)) {
        $rowsByStudent[$record.student_id] = [System.Collections.Generic.List[object]]::new()
    }
    $rowsByStudent[$record.student_id].Add($record) | Out-Null
}

$finalRows = [System.Collections.Generic.List[object]]::new()
foreach ($studentId in ($rowsByStudent.Keys | Sort-Object)) {
    $orderedRows = $rowsByStudent[$studentId] | Sort-Object term_order, school_year, semester
    for ($i = 0; $i -lt $orderedRows.Count; $i++) {
        $row = $orderedRows[$i]
        $nextTermRisk = if ($i -lt ($orderedRows.Count - 1)) { [string]$orderedRows[$i + 1].risk_event_current } else { "" }

        $avgScore = ""
        $scoreStd = ""
        if ($row.score_numeric_count -gt 0) {
            $mean = $row.score_sum / $row.score_numeric_count
            $avgScore = [math]::Round($mean, 4)
            if ($row.score_numeric_count -gt 1) {
                $variance = (($row.score_sq_sum / $row.score_numeric_count) - ($mean * $mean))
                if ($variance -lt 0) { $variance = 0 }
                $scoreStd = [math]::Round([math]::Sqrt($variance), 4)
            }
            else {
                $scoreStd = 0
            }
        }

        $avgGpa = if ($row.gpa_count -gt 0) { [math]::Round(($row.gpa_sum / $row.gpa_count), 4) } else { "" }
        $failRatio = if ($row.score_course_count -gt 0) { [math]::Round(($row.fail_course_count / $row.score_course_count), 4) } else { "" }
        $internetAvgPerMonth = if ($row.internet_month_count -gt 0) { [math]::Round(($row.internet_hours_sum / $row.internet_month_count), 4) } else { "" }
        $internetDiffMean = if ($row.internet_month_count -gt 0) { [math]::Round(($row.internet_diff_sum / $row.internet_month_count), 4) } else { "" }

        if ($onlineSnapshotMap.ContainsKey($studentId)) {
            $row.online_learning_bfb_snapshot = [math]::Round([double]$onlineSnapshotMap[$studentId], 4)
        }

        $physicalKey = "$studentId|$($row.school_year)"
        if ($physicalYearMap.ContainsKey($physicalKey) -and $physicalYearMap[$physicalKey].zf_count -gt 0) {
            $row.physical_test_score = [math]::Round(($physicalYearMap[$physicalKey].zf_sum / $physicalYearMap[$physicalKey].zf_count), 4)
            $row.physical_test_bmi = $physicalYearMap[$physicalKey].bmi_text
        }

        $finalRows.Add([pscustomobject]@{
            student_id                   = $row.student_id
            school_year                  = $row.school_year
            semester                     = $row.semester
            term_order                   = $row.term_order
            gender                       = $row.gender
            grade                        = $row.grade
            college                      = $row.college
            major                        = $row.major
            class_name                   = $row.class_name
            selected_course_count        = $row.selected_course_count
            score_course_count           = $row.score_course_count
            avg_score                    = $avgScore
            score_std                    = $scoreStd
            fail_course_count            = $row.fail_course_count
            fail_ratio                   = $failRatio
            avg_gpa                      = $avgGpa
            credit_sum                   = [math]::Round($row.credit_sum, 4)
            resit_exam_count             = $row.resit_exam_count
            internet_month_count         = $row.internet_month_count
            internet_hours_sum           = [math]::Round($row.internet_hours_sum, 4)
            internet_hours_avg_per_month = $internetAvgPerMonth
            internet_diff_mean           = $internetDiffMean
            online_learning_bfb_snapshot = $row.online_learning_bfb_snapshot
            physical_test_score          = $row.physical_test_score
            physical_test_bmi            = $row.physical_test_bmi
            risk_event_current           = $row.risk_event_current
            risk_event_type_codes        = Join-HashSet -Value $row.risk_event_type_codes
            risk_event_reason_codes      = Join-HashSet -Value $row.risk_event_reason_codes
            risk_label_next_term         = $nextTermRisk
        }) | Out-Null
    }
}

$finalRows = $finalRows | Sort-Object student_id, term_order
$finalRows | Export-Csv -LiteralPath ".\prepared\03_datasets\student_semester_base.csv" -NoTypeInformation -Encoding utf8
$finalRows | Export-Csv -LiteralPath ".\student_semester_base.csv" -NoTypeInformation -Encoding utf8

$reportLines = @(
    "# student_semester_base build report",
    "",
    ("- rows: {0}" -f $finalRows.Count),
    ("- unique_students: {0}" -f (($finalRows | Select-Object -ExpandProperty student_id -Unique).Count)),
    ("- current_risk_rows: {0}" -f (($finalRows | Where-Object { $_.risk_event_current -eq '1' }).Count)),
    ("- nonempty_next_term_label_rows: {0}" -f (($finalRows | Where-Object { $_.risk_label_next_term -ne '' }).Count))
)

$reportLines | Set-Content -LiteralPath ".\prepared\03_datasets\student_semester_base_build_report.md" -Encoding utf8

Write-Host "Step 5/5: done"
Write-Host "Outputs:"
Write-Host "  prepared\\01_keys\\student_master.csv"
Write-Host "  prepared\\03_datasets\\student_semester_base.csv"
