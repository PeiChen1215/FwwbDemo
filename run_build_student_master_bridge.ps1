$ErrorActionPreference = "Stop"

$helperFile = Get-ChildItem -LiteralPath "." -Filter "*.ps1" |
    Where-Object { $_.Name -ne "run_build_student_master_bridge.ps1" } |
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

        if (Headers-Contain -Headers $headers -Required @("XH", "XSM", "ZYM", "CSRQ", "JG", "MZMC")) { $aliases["student_basic"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XM", "XH", "CFBFS", "HSCJ", "XQ")) { $aliases["student_fitness"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "TCNF", "BMI", "FHL", "WS", "LDTY")) { $aliases["physical_test"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "ZKC", "KC", "SKSJ")) { $aliases["pe_class"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "ZC", "DKCS")) { $aliases["daily_exercise"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("SID", "GRADE", "BYQXMC")) { $aliases["graduation"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("LOGIN_NAME", "ROLEID", "BFB", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME")) { $aliases["online_learning"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("LOGIN_NAME", "USER_ID", "COURSE_ID", "JOB_NUM", "TASK_RATE")) { $aliases["class_task"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("LOGIN_NAME", "ACTIVE_ID", "ROLE", "XYMC", "MAJORNAME", "CLASSNAME")) { $aliases["sign_record"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("CREATER_LOGIN_NAME", "WORK_ID", "COMMENT", "SCORE", "TYPE")) { $aliases["homework_submit"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("CREATER_LOGIN_NAME", "WORK_ID", "FULLMARKS", "SCORE", "TYPE")) { $aliases["exam_submit"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("CREATER_LOGIN_NAME", "REPLY_LOGIN_NAME", "TOPIC_ID", "TOPIC_TITLE")) { $aliases["discussion"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "KCH", "KCCJ", "JDCJ")) { $aliases["student_score"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "JXBH", "KCH", "KKXN", "KKXQ")) { $aliases["course_select"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "YDRQ", "YDLBDM", "SFZX")) { $aliases["status_change"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XH", "RQ", "ZT", "DKSJ")) { $aliases["attendance_summary"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XSBH", "TJNY", "SWLJSC")) { $aliases["internet_stats"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XSBH", "JXJMC", "FFJE")) { $aliases["scholarship"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XSBH", "HDRQ")) { $aliases["club"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("USERNUM", "PUNCH_DAY", "TERM_ID")) { $aliases["running"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("IDSERTAL", "LOGINTIME", "LOGINADDRESS")) { $aliases["access"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("cardld", "visittime", "gateno")) { $aliases["library"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("KS_XH", "KS_YYJB", "KS_CJ")) { $aliases["cet"] = $file; continue }
        if (Headers-Contain -Headers $headers -Required @("XHHGH", "JSMC", "HJJB", "HJDJ")) { $aliases["competition"] = $file; continue }
    }

    $requiredAliases = @(
        "student_basic", "student_fitness", "physical_test", "pe_class", "daily_exercise", "graduation",
        "online_learning", "class_task", "sign_record", "homework_submit", "exam_submit", "discussion",
        "student_score", "course_select", "status_change", "attendance_summary", "internet_stats",
        "scholarship", "club", "running", "access", "library", "cet", "competition"
    )

    foreach ($alias in $requiredAliases) {
        if (-not $aliases.ContainsKey($alias)) {
            throw "Missing data file alias: $alias"
        }
    }

    return $aliases
}

$files = Resolve-DataFiles
Write-Host "Step 1/6: build student_master"

$studentMasterMap = @{}

foreach ($row in Get-XlsxRows -Path $files.student_basic.FullName -SelectColumns @("XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    foreach ($field in @("XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM")) {
        Set-FirstNonEmpty -Record $record -Field $field -Value $row.$field
    }
    Add-RecordSource -Record $record -SourceName $files.student_basic.Name
}

foreach ($item in @(
    @{ File = $files.student_fitness; Columns = @("XM", "XH", "XB", "NJ", "YX", "ZY", "BJ", "BH") },
    @{ File = $files.physical_test; Columns = @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH") },
    @{ File = $files.pe_class; Columns = @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH") },
    @{ File = $files.daily_exercise; Columns = @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH") }
)) {
    foreach ($row in Get-XlsxRows -Path $item.File.FullName -SelectColumns $item.Columns) {
        $studentId = Normalize-Id -Value $row.XH
        if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
        $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
        foreach ($field in @("XM", "XB", "NJ", "YX", "ZY", "BJ", "BH")) {
            if ($item.Columns -contains $field) {
                Set-FirstNonEmpty -Record $record -Field $field -Value $row.$field
            }
        }
        Add-RecordSource -Record $record -SourceName $item.File.Name
    }
}

foreach ($row in Get-XlsxRows -Path $files.graduation.FullName -SelectColumns @("SID", "GRADE", "BJMC", "BYQXMC")) {
    $studentId = Normalize-Id -Value $row.SID
    if ([string]::IsNullOrWhiteSpace($studentId)) { continue }
    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "GRADE" -Value $row.GRADE
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJMC
    Set-FirstNonEmpty -Record $record -Field "BYQXMC" -Value $row.BYQXMC
    Add-RecordSource -Record $record -SourceName $files.graduation.Name
}

$studentMasterRows = [System.Collections.Generic.List[object]]::new()
$orgClassIndex = @{}
$orgMajorIndex = @{}

foreach ($studentId in ($studentMasterMap.Keys | Sort-Object)) {
    $record = $studentMasterMap[$studentId]
    if ([string]::IsNullOrWhiteSpace($record.YX)) { $record.YX = $record.XSM }
    if ([string]::IsNullOrWhiteSpace($record.ZY)) { $record.ZY = $record.ZYM }
    if ([string]::IsNullOrWhiteSpace($record.BJ)) { $record.BJ = $record.BH }
    $orgClassKey = New-Key -Parts @($record.YX, $record.ZY, $record.BJ)
    $orgMajorKey = New-Key -Parts @($record.YX, $record.ZY)
    Add-ToKeyIndex -Index $orgClassIndex -Key $orgClassKey -StudentId $studentId
    Add-ToKeyIndex -Index $orgMajorIndex -Key $orgMajorKey -StudentId $studentId
    $studentMasterRows.Add([pscustomobject]@{
        student_id      = $studentId
        XM              = $record.XM
        XB              = $record.XB
        NJ              = $record.NJ
        YX              = $record.YX
        ZY              = $record.ZY
        BJ              = $record.BJ
        BH              = $record.BH
        XSM             = $record.XSM
        ZYM             = $record.ZYM
        MZMC            = $record.MZMC
        ZZMMMC          = $record.ZZMMMC
        CSRQ            = $record.CSRQ
        JG              = $record.JG
        GRADE           = $record.GRADE
        BYQXMC          = $record.BYQXMC
        department_norm = Normalize-Text -Value $record.YX
        major_norm      = Normalize-Text -Value $record.ZY
        class_norm      = Normalize-Text -Value $record.BJ
        org_class_key   = $orgClassKey
        org_major_key   = $orgMajorKey
        source_tables   = Join-Set -Set $record.source_tables
    }) | Out-Null
}

Write-Host "Step 2/6: build account_master"

$accountMasterMap = @{}
$accountSourceValues = @{}

foreach ($row in Get-XlsxRows -Path $files.online_learning.FullName -SelectColumns @("LOGIN_NAME", "XM", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME", "ROLEID")) {
    if (($row.ROLEID).Trim() -ne "3") { continue }
    $loginRaw = [string]$row.LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) { continue }
    Register-SourceValue -ValueStore $accountSourceValues -SourceTable $files.online_learning.Name -SourceField "LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "user_name" -Value $row.XM
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.DEPARTMENT_NAME
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJOR_NAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASS_NAME
    Add-RecordSource -Record $record -SourceName $files.online_learning.Name
}

Write-Host "Step 3/6: resolve account mapping"

foreach ($loginNorm in $accountMasterMap.Keys) {
    $record = $accountMasterMap[$loginNorm]

    if ($studentMasterMap.ContainsKey($loginNorm)) {
        $record.student_id = $loginNorm
        $record.match_type = "exact_login_alias"
        $record.match_confidence = "high"
        $record.evidence_fields = "login_name = student_id"
        continue
    }

    $orgClassKey = New-Key -Parts @($record.department_name, $record.major_name, $record.class_name)
    if (-not [string]::IsNullOrWhiteSpace($orgClassKey) -and $orgClassIndex.ContainsKey($orgClassKey) -and $orgClassIndex[$orgClassKey].Count -eq 1) {
        $record.student_id = @($orgClassIndex[$orgClassKey])[0]
        $record.match_type = "bridge_org_class_unique"
        $record.match_confidence = "medium"
        $record.evidence_fields = "department_name + major_name + class_name"
        continue
    }

    $orgMajorKey = New-Key -Parts @($record.department_name, $record.major_name)
    if (-not [string]::IsNullOrWhiteSpace($orgMajorKey) -and $orgMajorIndex.ContainsKey($orgMajorKey) -and $orgMajorIndex[$orgMajorKey].Count -eq 1) {
        $record.student_id = @($orgMajorIndex[$orgMajorKey])[0]
        $record.match_type = "bridge_org_major_unique"
        $record.match_confidence = "low"
        $record.evidence_fields = "department_name + major_name"
        continue
    }

    $record.student_id = ""
    $record.match_type = "unresolved_org_multi"
    $record.match_confidence = "low"
    $record.evidence_fields = "organization tuple not unique after anonymization"
}

$accountMasterRows = [System.Collections.Generic.List[object]]::new()
foreach ($loginNorm in ($accountMasterMap.Keys | Sort-Object)) {
    $record = $accountMasterMap[$loginNorm]
    $accountMasterRows.Add([pscustomobject]@{
        login_name_raw   = $record.login_name_raw
        login_name_norm  = $record.login_name_norm
        user_name        = $record.user_name
        department_name  = $record.department_name
        major_name       = $record.major_name
        class_name       = $record.class_name
        student_id       = $record.student_id
        match_type       = $record.match_type
        match_confidence = $record.match_confidence
        evidence_fields  = $record.evidence_fields
        source_tables    = Join-Set -Set $record.source_tables
    }) | Out-Null
}

Write-Host "Step 4/6: build direct bridge"

$bridgeRows = [System.Collections.Generic.List[object]]::new()
$identityMappings = @(
    @{ File = $files.student_basic;      Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.student_score;      Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.course_select;      Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.status_change;      Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.daily_exercise;     Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.pe_class;           Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.physical_test;      Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.student_fitness;    Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.attendance_summary; Field = "XH";    Evidence = "identity:XH -> student_id" },
    @{ File = $files.internet_stats;     Field = "XSBH";  Evidence = "identity:XSBH -> student_id" },
    @{ File = $files.scholarship;        Field = "XSBH";  Evidence = "identity:XSBH -> student_id" },
    @{ File = $files.club;               Field = "XSBH";  Evidence = "identity:XSBH -> student_id" },
    @{ File = $files.graduation;         Field = "SID";   Evidence = "identity:SID -> student_id" }
)

foreach ($mapping in $identityMappings) {
    Write-Host ("  - template {0}::{1}" -f $mapping.File.Name, $mapping.Field)
    foreach ($studentId in ($studentMasterMap.Keys | Sort-Object)) {
        Add-BridgeRow -BridgeRows $bridgeRows -SourceTable $mapping.File.Name -SourceField $mapping.Field -SourceValueRaw $studentId -SourceValueNorm $studentId -StudentId $studentId -MatchType "exact" -MatchConfidence "high" -EvidenceFields $mapping.Evidence
    }
}

Write-Host ("  - scan {0}::{1}" -f $files.competition.Name, "XHHGH")
$competitionValues = (Get-UniqueValueMap -Path $files.competition.FullName -Columns @("XHHGH"))["XHHGH"]
foreach ($valueNorm in ($competitionValues.Keys | Sort-Object)) {
    $valueRaw = $competitionValues[$valueNorm]
    if ($studentMasterMap.ContainsKey($valueNorm)) {
        Add-BridgeRow -BridgeRows $bridgeRows -SourceTable $files.competition.Name -SourceField "XHHGH" -SourceValueRaw $valueRaw -SourceValueNorm $valueNorm -StudentId $valueNorm -MatchType "filtered_exact" -MatchConfidence "high" -EvidenceFields "XHHGH filtered by student_master"
    }
    else {
        Add-BridgeRow -BridgeRows $bridgeRows -SourceTable $files.competition.Name -SourceField "XHHGH" -SourceValueRaw $valueRaw -SourceValueNorm $valueNorm -StudentId "" -MatchType "filtered_out_nonstudent" -MatchConfidence "none" -EvidenceFields "XHHGH filtered by student_master"
    }
}

Write-Host "Step 5/6: add account bridge rows"

foreach ($storeKey in ($accountSourceValues.Keys | Sort-Object)) {
    $parts = $storeKey -split "\|", 2
    $sourceTable = $parts[0]
    $sourceField = $parts[1]
    $valueMap = $accountSourceValues[$storeKey]

    foreach ($valueNorm in ($valueMap.Keys | Sort-Object)) {
        $valueRaw = $valueMap[$valueNorm]
        $studentId = ""
        $matchType = "unresolved_missing_account_master"
        $matchConfidence = "none"
        $evidenceFields = "login_name"

        if ($accountMasterMap.ContainsKey($valueNorm)) {
            $accountRecord = $accountMasterMap[$valueNorm]
            $studentId = $accountRecord.student_id
            $matchType = $accountRecord.match_type
            $matchConfidence = $accountRecord.match_confidence
            $evidenceFields = $accountRecord.evidence_fields
        }

        Add-BridgeRow -BridgeRows $bridgeRows -SourceTable $sourceTable -SourceField $sourceField -SourceValueRaw $valueRaw -SourceValueNorm $valueNorm -StudentId $studentId -MatchType $matchType -MatchConfidence $matchConfidence -EvidenceFields $evidenceFields
    }
}

Write-Host "Step 6/6: export files"

$studentMasterOutput = $studentMasterRows | Sort-Object student_id
$accountMasterOutput = $accountMasterRows | Sort-Object login_name_norm
$bridgeOutput = $bridgeRows | Sort-Object source_table, source_field, source_value_norm
$bridgeSummaryOutput = $bridgeOutput |
    Group-Object source_table, source_field, match_type |
    ForEach-Object {
        [pscustomobject]@{
            source_table = $_.Group[0].source_table
            source_field = $_.Group[0].source_field
            match_type   = $_.Group[0].match_type
            row_count    = $_.Count
        }
    } |
    Sort-Object source_table, source_field, match_type

$studentMasterOutput | Export-Csv -LiteralPath ".\student_master.csv" -NoTypeInformation -Encoding utf8
$accountMasterOutput | Export-Csv -LiteralPath ".\account_master.csv" -NoTypeInformation -Encoding utf8
$bridgeOutput | Export-Csv -LiteralPath ".\student_key_bridge.csv" -NoTypeInformation -Encoding utf8
$bridgeSummaryOutput | Export-Csv -LiteralPath ".\student_key_bridge_summary.csv" -NoTypeInformation -Encoding utf8

$reportLines = @(
    "# student_master and bridge build report",
    "",
    ("- student_master rows: {0}" -f $studentMasterOutput.Count),
    ("- account_master rows: {0}" -f $accountMasterOutput.Count),
    ("- student_key_bridge rows: {0}" -f $bridgeOutput.Count),
    "",
    "## notes",
    "",
    "- direct id fields are exported as exact identity templates or filtered_exact matches",
    "- account fields are first aggregated into account_master, then mapped only when the organization tuple is uniquely identifiable",
    "- unresolved account rows are kept unresolved instead of being force-matched",
    "- this first bridge export focuses on the high-priority exact-id tables plus online-learning accounts",
    "- large account event tables can be added in a later optimized pass if needed",
    "",
    "## output files",
    "",
    "- student_master.csv",
    "- account_master.csv",
    "- student_key_bridge.csv",
    "- student_key_bridge_summary.csv"
)

Set-Content -LiteralPath ".\student_master_bridge_build_report.md" -Value $reportLines -Encoding utf8

Write-Host "done: student_master.csv / account_master.csv / student_key_bridge.csv / student_key_bridge_summary.csv"
