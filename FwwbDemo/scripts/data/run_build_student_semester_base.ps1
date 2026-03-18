$ErrorActionPreference = "Stop"

$helperFile = Get-ChildItem -LiteralPath "." -Filter "*.ps1" |
    Where-Object { $_.Name -ne $MyInvocation.MyCommand.Name } |
    Sort-Object Length -Descending |
    Select-Object -First 1

if ($null -eq $helperFile) {
    throw "No helper ps1 file found."
}

$helperLines = Get-Content -LiteralPath $helperFile.FullName
$splitIndex = [Array]::FindIndex($helperLines, [Predicate[string]]{
    param($line)
    $line -like 'Write-Host "Step 1/6*'
})

if ($splitIndex -lt 0) {
    throw "Unable to locate helper function boundary."
}

$helperText = ($helperLines[0..($splitIndex - 1)] -join "`r`n")
Invoke-Expression $helperText

function Headers-Contain {
    param([string[]]$Headers, [string[]]$Required)

    $set = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($header in $Headers) {
        [void]$set.Add($header)
    }

    foreach ($item in $Required) {
        if (-not $set.Contains($item)) {
            return $false
        }
    }

    return $true
}

function Get-HeaderNames {
    param([string]$Path)

    $firstRow = Get-XlsxRows -Path $Path | Select-Object -First 1
    if ($null -eq $firstRow) {
        return @()
    }

    return @($firstRow.PSObject.Properties.Name)
}

function Resolve-DataFiles {
    $aliases = @{}
    $xlsxFiles = Get-ChildItem -LiteralPath "." -Filter "*.xlsx" | Where-Object { $_.Name -notlike '~$*' }

    foreach ($file in $xlsxFiles) {
        $headers = Get-HeaderNames -Path $file.FullName

        if (Headers-Contain -Headers $headers -Required @("XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM")) { $aliases["student_basic"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH", "TCNF", "ZF")) { $aliases["physical_test"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("LOGIN_NAME", "XM", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME", "ROLEID", "BFB")) { $aliases["online_learning"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "JXBH", "KCH", "KKXN", "KKXQ")) { $aliases["course_select"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "ZXJXJHH", "KCH", "KXH", "KCCJ", "JDCJ")) { $aliases["student_score"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XSBH", "TJNY", "SWLJSC")) { $aliases["internet_stats"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "YDRQ", "YDLBDM", "YDYYDM", "SPRQ", "SFZX")) { $aliases["status_change"] = $file; continue }
    }

    $requiredAliases = @("student_basic", "physical_test", "online_learning", "course_select", "student_score", "internet_stats", "status_change")
    foreach ($alias in $requiredAliases) {
        if (-not $aliases.ContainsKey($alias)) {
            throw "Missing required file alias: $alias"
        }
    }

    return $aliases
}

function To-DoubleOrNull {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }

    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    $number = 0.0
    if ([double]::TryParse($text, [ref]$number)) {
        return $number
    }

    return $null
}

function Convert-ToDate {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }

    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    if ($text -match '^\d{8}$') {
        try {
            return [datetime]::ParseExact($text, 'yyyyMMdd', $null)
        }
        catch {
            return $null
        }
    }

    $asDouble = 0.0
    if ([double]::TryParse($text, [ref]$asDouble)) {
        try {
            return [datetime]::FromOADate($asDouble)
        }
        catch {
            return $null
        }
    }

    try {
        return [datetime]$text
    }
    catch {
        return $null
    }
}

function Get-SchoolYearSemesterFromDate {
    param([datetime]$Date)

    if ($null -eq $Date) {
        return $null
    }

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

    if ([string]::IsNullOrWhiteSpace($PlanValue)) {
        return $null
    }

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

    if ([string]::IsNullOrWhiteSpace($SchoolYear) -or [string]::IsNullOrWhiteSpace($Semester)) {
        return -1
    }

    $startYear = 0
    if ($SchoolYear -match '^(?<start>\d{4})-\d{4}$') {
        $startYear = [int]$Matches["start"]
    }

    return ($startYear * 10 + [int]$Semester)
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
    param([hashtable]$Record, [string]$Field, [object]$Value)

    $text = ""
    if ($null -ne $Value) {
        $text = ([string]$Value).Trim()
    }

    if ([string]::IsNullOrWhiteSpace($text)) {
        return
    }

    if ([string]::IsNullOrWhiteSpace([string]$Record[$Field])) {
        $Record[$Field] = $text
    }
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
        $profile = $StudentProfiles[$StudentId]
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
            internet_school_avg_sum       = 0.0
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

    if ($null -eq $Value) {
        return ""
    }

    return (($Value | Sort-Object) -join "|")
}

$files = Resolve-DataFiles

Write-Host "Step 1/5: rebuild student_master from source files"

$studentProfiles = @{}

foreach ($row in Get-XlsxRows -Path $files.student_basic.FullName -SelectColumns @("XH", "XB", "XSM", "ZYM")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "XB" -Value $row.XB
    Set-IfEmpty -Record $profile -Field "XSM" -Value $row.XSM
    Set-IfEmpty -Record $profile -Field "ZYM" -Value $row.ZYM
}

foreach ($row in Get-XlsxRows -Path $files.physical_test.FullName -SelectColumns @("XH", "XB", "NJ", "YX", "ZY", "BJ")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "XB" -Value $row.XB
    Set-IfEmpty -Record $profile -Field "NJ" -Value $row.NJ
    Set-IfEmpty -Record $profile -Field "YX" -Value $row.YX
    Set-IfEmpty -Record $profile -Field "ZY" -Value $row.ZY
    Set-IfEmpty -Record $profile -Field "BJ" -Value $row.BJ
}

foreach ($row in Get-XlsxRows -Path $files.online_learning.FullName -SelectColumns @("LOGIN_NAME", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME", "ROLEID")) {
    if (($row.ROLEID).Trim() -ne "3") { continue }
    $studentId = Normalize-Id -Value $row.LOGIN_NAME
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $profile = Get-OrCreateStudentProfile -Map $studentProfiles -StudentId $studentId
    Set-IfEmpty -Record $profile -Field "YX" -Value $row.DEPARTMENT_NAME
    Set-IfEmpty -Record $profile -Field "ZY" -Value $row.MAJOR_NAME
    Set-IfEmpty -Record $profile -Field "BJ" -Value $row.CLASS_NAME
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

Write-Host "Step 2/5: build student-semester skeleton"

$semesterMap = @{}

foreach ($row in Get-XlsxRows -Path $files.course_select.FullName -SelectColumns @("XH", "KKXN", "KKXQ")) {
    $studentId = Normalize-Id -Value $row.XH
    $schoolYear = ([string]$row.KKXN).Trim()
    $semester = ([string]$row.KKXQ).Trim()
    if ([string]::IsNullOrWhiteSpace($studentId) -or [string]::IsNullOrWhiteSpace($schoolYear) -or [string]::IsNullOrWhiteSpace($semester)) { continue }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $schoolYear -Semester $semester
    $record.selected_course_count += 1
}

foreach ($row in Get-XlsxRows -Path $files.student_score.FullName -SelectColumns @("XH", "ZXJXJHH")) {
    $studentId = Normalize-Id -Value $row.XH
    $termInfo = Get-TermInfoFromScorePlan -PlanValue ([string]$row.ZXJXJHH)
    if ([string]::IsNullOrWhiteSpace($studentId) -or $null -eq $termInfo) { continue }
    [void](Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester)
}

Write-Host "Step 3/5: aggregate score, internet, online-learning, physical-test features"

foreach ($row in Get-XlsxRows -Path $files.student_score.FullName -SelectColumns @("XH", "ZXJXJHH", "XF", "CXBKBZ", "KCCJ", "JDCJ")) {
    $studentId = Normalize-Id -Value $row.XH
    $termInfo = Get-TermInfoFromScorePlan -PlanValue ([string]$row.ZXJXJHH)
    if ([string]::IsNullOrWhiteSpace($studentId) -or $null -eq $termInfo) { continue }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester

    $record.score_course_count += 1
    $scoreValue = To-DoubleOrNull -Value $row.KCCJ
    if ($null -ne $scoreValue) {
        $record.score_numeric_count += 1
        $record.score_sum += $scoreValue
        $record.score_sq_sum += ($scoreValue * $scoreValue)
        if ($scoreValue -lt 60) {
            $record.fail_course_count += 1
        }
    }

    $gpaValue = To-DoubleOrNull -Value $row.JDCJ
    if ($null -ne $gpaValue) {
        $record.gpa_count += 1
        $record.gpa_sum += $gpaValue
    }

    $creditValue = To-DoubleOrNull -Value $row.XF
    if ($null -ne $creditValue) {
        $record.credit_sum += $creditValue
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$row.CXBKBZ) -and ([string]$row.CXBKBZ).Trim() -ne "01") {
        $record.resit_exam_count += 1
    }
}

foreach ($row in Get-XlsxRows -Path $files.internet_stats.FullName -SelectColumns @("XSBH", "TJNY", "SWLJSC", "XXPJZ")) {
    $studentId = Normalize-Id -Value $row.XSBH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $dateValue = Convert-ToDate -Value $row.TJNY
    $termInfo = Get-SchoolYearSemesterFromDate -Date $dateValue
    if ($null -eq $termInfo) { continue }
    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester

    $record.internet_month_count += 1
    $hoursValue = To-DoubleOrNull -Value $row.SWLJSC
    if ($null -ne $hoursValue) {
        $record.internet_hours_sum += $hoursValue
    }

    $schoolAvgValue = To-DoubleOrNull -Value $row.XXPJZ
    if ($null -ne $schoolAvgValue) {
        $record.internet_school_avg_sum += $schoolAvgValue
        if ($null -ne $hoursValue) {
            $record.internet_diff_sum += ($hoursValue - $schoolAvgValue)
        }
    }
}

$onlineSnapshotMap = @{}
foreach ($row in Get-XlsxRows -Path $files.online_learning.FullName -SelectColumns @("LOGIN_NAME", "ROLEID", "BFB")) {
    if (($row.ROLEID).Trim() -ne "3") { continue }
    $studentId = Normalize-Id -Value $row.LOGIN_NAME
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $bfbValue = To-DoubleOrNull -Value $row.BFB
    if ($null -ne $bfbValue -and -not $onlineSnapshotMap.ContainsKey($studentId)) {
        $onlineSnapshotMap[$studentId] = $bfbValue
    }
}

$physicalYearMap = @{}
foreach ($row in Get-XlsxRows -Path $files.physical_test.FullName -SelectColumns @("XH", "TCNF", "ZF", "BMI")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $yearText = ([string]$row.TCNF).Trim()
    if ($yearText -notmatch '^\d{4}$') { continue }
    $schoolYear = ("{0}-{1}" -f $yearText, ([int]$yearText + 1))
    $key = "$studentId|$schoolYear"
    if (-not $physicalYearMap.ContainsKey($key)) {
        $physicalYearMap[$key] = [ordered]@{
            zf_sum    = 0.0
            zf_count  = 0
            bmi_text  = ""
        }
    }
    $zfValue = To-DoubleOrNull -Value $row.ZF
    if ($null -ne $zfValue) {
        $physicalYearMap[$key].zf_sum += $zfValue
        $physicalYearMap[$key].zf_count += 1
    }
    if ([string]::IsNullOrWhiteSpace([string]$physicalYearMap[$key].bmi_text) -and -not [string]::IsNullOrWhiteSpace([string]$row.BMI)) {
        $physicalYearMap[$key].bmi_text = ([string]$row.BMI).Trim()
    }
}

Write-Host "Step 4/5: derive current-term risk events and next-term labels"

$highRiskTypeSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($riskType in @("1", "3", "11", "13")) {
    [void]$highRiskTypeSet.Add($riskType)
}

foreach ($row in Get-XlsxRows -Path $files.status_change.FullName -SelectColumns @("XH", "YDRQ", "YDLBDM", "YDYYDM", "SPRQ", "BY1", "SFZX")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }

    $typeCode = ([string]$row.YDLBDM).Trim()
    $reasonCode = ([string]$row.YDYYDM).Trim()
    $sfzx = ([string]$row.SFZX).Trim()
    $isHighRisk = $highRiskTypeSet.Contains($typeCode) -or $sfzx -eq "0"
    if (-not $isHighRisk) { continue }

    $eventDate = Convert-ToDate -Value $row.YDRQ
    if ($null -eq $eventDate) { $eventDate = Convert-ToDate -Value $row.SPRQ }
    if ($null -eq $eventDate) { $eventDate = Convert-ToDate -Value $row.BY1 }
    $termInfo = Get-SchoolYearSemesterFromDate -Date $eventDate
    if ($null -eq $termInfo) { continue }

    $record = Get-OrCreateSemesterRecord -Map $semesterMap -StudentProfiles $studentProfiles -StudentId $studentId -SchoolYear $termInfo.school_year -Semester $termInfo.semester
    $record.risk_event_current = 1
    if (-not [string]::IsNullOrWhiteSpace($typeCode)) {
        [void]$record.risk_event_type_codes.Add($typeCode)
    }
    if (-not [string]::IsNullOrWhiteSpace($reasonCode)) {
        [void]$record.risk_event_reason_codes.Add($reasonCode)
    }
}

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
        $nextTermRisk = ""
        if ($i -lt ($orderedRows.Count - 1)) {
            $nextTermRisk = [string]$orderedRows[$i + 1].risk_event_current
        }

        $avgScore = ""
        $scoreStd = ""
        if ($row.score_numeric_count -gt 0) {
            $avgScore = [math]::Round(($row.score_sum / $row.score_numeric_count), 4)
            if ($row.score_numeric_count -gt 1) {
                $variance = (($row.score_sq_sum / $row.score_numeric_count) - [math]::Pow(($row.score_sum / $row.score_numeric_count), 2))
                if ($variance -lt 0) { $variance = 0 }
                $scoreStd = [math]::Round([math]::Sqrt($variance), 4)
            }
            else {
                $scoreStd = 0
            }
        }

        $avgGpa = ""
        if ($row.gpa_count -gt 0) {
            $avgGpa = [math]::Round(($row.gpa_sum / $row.gpa_count), 4)
        }

        $failRatio = ""
        if ($row.score_course_count -gt 0) {
            $failRatio = [math]::Round(($row.fail_course_count / $row.score_course_count), 4)
        }

        $internetAvgPerMonth = ""
        $internetDiffMean = ""
        if ($row.internet_month_count -gt 0) {
            $internetAvgPerMonth = [math]::Round(($row.internet_hours_sum / $row.internet_month_count), 4)
            $internetDiffMean = [math]::Round(($row.internet_diff_sum / $row.internet_month_count), 4)
        }

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
    ("- nonempty_next_term_label_rows: {0}" -f (($finalRows | Where-Object { $_.risk_label_next_term -ne '' }).Count)),
    "",
    "Columns include:",
    "- static student profile fields",
    "- score aggregates",
    "- internet usage aggregates",
    "- online learning snapshot feature",
    "- physical test yearly feature",
    "- current-term risk event flag",
    "- next-term risk label"
)

$reportLines | Set-Content -LiteralPath ".\prepared\03_datasets\student_semester_base_build_report.md" -Encoding utf8

Write-Host "Step 5/5: done"
Write-Host "Outputs:"
Write-Host "  prepared\\01_keys\\student_master.csv"
Write-Host "  prepared\\03_datasets\\student_semester_base.csv"
