$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Normalize-Text {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }

    $text = [string]$Value
    $text = $text.Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return ""
    }

    return (($text -replace "\s+", "")).ToLowerInvariant()
}

function Normalize-Id {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }

    $text = [string]$Value
    $text = $text.Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return ""
    }

    return $text.ToLowerInvariant()
}

function Join-Set {
    param([object]$Set)

    if ($null -eq $Set) {
        return ""
    }

    return (($Set | Sort-Object) -join "|")
}

function Get-ColumnLetters {
    param([string]$CellReference)

    if ([string]::IsNullOrWhiteSpace($CellReference)) {
        return ""
    }

    return [regex]::Match($CellReference, "^[A-Z]+").Value
}

function Get-XlsxSharedStrings {
    param([System.IO.Compression.ZipArchive]$Zip)

    $entry = $Zip.GetEntry("xl/sharedStrings.xml")
    $strings = [System.Collections.Generic.List[string]]::new()
    if ($null -eq $entry) {
        return $strings
    }

    $reader = [System.Xml.XmlReader]::Create($entry.Open())
    try {
        while ($reader.Read()) {
            if (
                $reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and
                $reader.LocalName -eq "si"
            ) {
                $subReader = $reader.ReadSubtree()
                $builder = [System.Text.StringBuilder]::new()
                try {
                    while ($subReader.Read()) {
                        if (
                            $subReader.NodeType -eq [System.Xml.XmlNodeType]::Element -and
                            $subReader.LocalName -eq "t"
                        ) {
                            [void]$builder.Append($subReader.ReadElementContentAsString())
                        }
                    }
                }
                finally {
                    $subReader.Dispose()
                }

                $strings.Add($builder.ToString())
            }
        }
    }
    finally {
        $reader.Dispose()
    }

    return $strings
}

function Read-XlsxCellValue {
    param(
        [System.Xml.XmlReader]$Reader,
        [System.Collections.Generic.List[string]]$SharedStrings
    )

    $cellType = $Reader.GetAttribute("t")
    $depth = $Reader.Depth
    $value = ""

    if ($Reader.IsEmptyElement) {
        return $value
    }

    while ($Reader.Read()) {
        if ($Reader.NodeType -eq [System.Xml.XmlNodeType]::Element) {
            if ($Reader.LocalName -eq "v") {
                $rawValue = $Reader.ReadElementContentAsString()
                if ($cellType -eq "s" -and $rawValue -match "^\d+$") {
                    return $SharedStrings[[int]$rawValue]
                }

                return $rawValue
            }

            if ($Reader.LocalName -eq "t") {
                $value += $Reader.ReadElementContentAsString()
            }
        }

        if (
            $Reader.NodeType -eq [System.Xml.XmlNodeType]::EndElement -and
            $Reader.LocalName -eq "c" -and
            $Reader.Depth -eq $depth
        ) {
            break
        }
    }

    return $value
}

function Get-XlsxRows {
    param(
        [string]$Path,
        [string[]]$SelectColumns = @()
    )

    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $sharedStrings = Get-XlsxSharedStrings -Zip $zip
        $sheetEntry = $zip.GetEntry("xl/worksheets/sheet1.xml")
        if ($null -eq $sheetEntry) {
            throw "鏈壘鍒?sheet1.xml: $Path"
        }

        $settings = [System.Xml.XmlReaderSettings]::new()
        $settings.IgnoreWhitespace = $true
        $reader = [System.Xml.XmlReader]::Create($sheetEntry.Open(), $settings)

        $headerByColumn = [ordered]@{}
        $selectedColumnNameSet = $null
        $selectedColumnLetterSet = $null

        if ($SelectColumns.Count -gt 0) {
            $selectedColumnNameSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
            foreach ($columnName in $SelectColumns) {
                if (-not [string]::IsNullOrWhiteSpace($columnName)) {
                    [void]$selectedColumnNameSet.Add($columnName)
                }
            }
        }

        try {
            while ($reader.Read()) {
                if (
                    $reader.NodeType -ne [System.Xml.XmlNodeType]::Element -or
                    $reader.LocalName -ne "row"
                ) {
                    continue
                }

                $cells = @{}
                $rowReader = $reader.ReadSubtree()
                try {
                    while ($rowReader.Read()) {
                        if (
                            $rowReader.NodeType -ne [System.Xml.XmlNodeType]::Element -or
                            $rowReader.LocalName -ne "c"
                        ) {
                            continue
                        }

                        $columnLetter = Get-ColumnLetters -CellReference $rowReader.GetAttribute("r")
                        if ([string]::IsNullOrWhiteSpace($columnLetter)) {
                            $rowReader.Skip()
                            continue
                        }

                        if ($selectedColumnLetterSet -and -not $selectedColumnLetterSet.Contains($columnLetter)) {
                            $rowReader.Skip()
                            continue
                        }

                        $cells[$columnLetter] = Read-XlsxCellValue -Reader $rowReader -SharedStrings $sharedStrings
                    }
                }
                finally {
                    $rowReader.Dispose()
                }

                if ($headerByColumn.Count -eq 0) {
                    foreach ($columnLetter in $cells.Keys) {
                        $headerByColumn[$columnLetter] = [string]$cells[$columnLetter]
                    }

                    if ($selectedColumnNameSet) {
                        $selectedColumnLetterSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
                        foreach ($columnLetter in $headerByColumn.Keys) {
                            $columnName = $headerByColumn[$columnLetter]
                            if ($selectedColumnNameSet.Contains($columnName)) {
                                [void]$selectedColumnLetterSet.Add($columnLetter)
                            }
                        }
                    }

                    continue
                }

                $record = [ordered]@{}
                foreach ($columnLetter in $headerByColumn.Keys) {
                    $columnName = $headerByColumn[$columnLetter]
                    if ([string]::IsNullOrWhiteSpace($columnName)) {
                        continue
                    }

                    if ($selectedColumnNameSet -and -not $selectedColumnNameSet.Contains($columnName)) {
                        continue
                    }

                    if ($cells.ContainsKey($columnLetter)) {
                        $record[$columnName] = [string]$cells[$columnLetter]
                    }
                    else {
                        $record[$columnName] = ""
                    }
                }

                if ($record.Count -gt 0) {
                    [pscustomobject]$record
                }
            }
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $zip.Dispose()
    }
}

function Set-FirstNonEmpty {
    param(
        [hashtable]$Record,
        [string]$Field,
        [object]$Value
    )

    $text = ""
    if ($null -ne $Value) {
        $text = [string]$Value
    }

    $text = $text.Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return
    }

    if ([string]::IsNullOrWhiteSpace([string]$Record[$Field])) {
        $Record[$Field] = $text
    }
}

function Add-RecordSource {
    param(
        [hashtable]$Record,
        [string]$SourceName
    )

    if (-not $Record.Contains("source_tables")) {
        $Record["source_tables"] = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    }

    [void]$Record["source_tables"].Add($SourceName)
}

function Get-OrCreateStudentRecord {
    param(
        [hashtable]$StudentMap,
        [string]$StudentId
    )

    if (-not $StudentMap.ContainsKey($StudentId)) {
        $StudentMap[$StudentId] = [ordered]@{
            student_id    = $StudentId
            XM            = ""
            XB            = ""
            NJ            = ""
            YX            = ""
            ZY            = ""
            BJ            = ""
            BH            = ""
            XSM           = ""
            ZYM           = ""
            MZMC          = ""
            ZZMMMC        = ""
            CSRQ          = ""
            JG            = ""
            GRADE         = ""
            BYQXMC        = ""
            source_tables = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        }
    }

    return $StudentMap[$StudentId]
}

function Get-OrCreateAccountRecord {
    param(
        [hashtable]$AccountMap,
        [string]$LoginNorm,
        [string]$LoginRaw
    )

    if (-not $AccountMap.ContainsKey($LoginNorm)) {
        $AccountMap[$LoginNorm] = [ordered]@{
            login_name_raw   = $LoginRaw
            login_name_norm  = $LoginNorm
            user_name        = ""
            department_name  = ""
            major_name       = ""
            class_name       = ""
            student_id       = ""
            match_type       = "unresolved"
            match_confidence = "low"
            evidence_fields  = ""
            source_tables    = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        }
    }

    return $AccountMap[$LoginNorm]
}

function New-Key {
    param([string[]]$Parts)

    $normalized = @()
    foreach ($part in $Parts) {
        $value = Normalize-Text -Value $part
        if ([string]::IsNullOrWhiteSpace($value)) {
            return ""
        }

        $normalized += $value
    }

    return ($normalized -join "|")
}

function Add-ToKeyIndex {
    param(
        [hashtable]$Index,
        [string]$Key,
        [string]$StudentId
    )

    if ([string]::IsNullOrWhiteSpace($Key)) {
        return
    }

    if (-not $Index.ContainsKey($Key)) {
        $Index[$Key] = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    }

    [void]$Index[$Key].Add($StudentId)
}

function Register-SourceValue {
    param(
        [hashtable]$ValueStore,
        [string]$SourceTable,
        [string]$SourceField,
        [string]$RawValue
    )

    $valueNorm = Normalize-Id -Value $RawValue
    if ([string]::IsNullOrWhiteSpace($valueNorm)) {
        return
    }

    $storeKey = "$SourceTable|$SourceField"
    if (-not $ValueStore.ContainsKey($storeKey)) {
        $ValueStore[$storeKey] = @{}
    }

    if (-not $ValueStore[$storeKey].ContainsKey($valueNorm)) {
        $ValueStore[$storeKey][$valueNorm] = $RawValue.Trim()
    }
}

function Add-BridgeRow {
    param(
        [System.Collections.Generic.List[object]]$BridgeRows,
        [string]$SourceTable,
        [string]$SourceField,
        [string]$SourceValueRaw,
        [string]$SourceValueNorm,
        [string]$StudentId,
        [string]$MatchType,
        [string]$MatchConfidence,
        [string]$EvidenceFields
    )

    $BridgeRows.Add([pscustomobject]@{
        source_table      = $SourceTable
        source_field      = $SourceField
        source_value_raw  = $SourceValueRaw
        source_value_norm = $SourceValueNorm
        student_id        = $StudentId
        match_type        = $MatchType
        match_confidence  = $MatchConfidence
        evidence_fields   = $EvidenceFields
    }) | Out-Null
}

function Get-UniqueValueMap {
    param(
        [string]$Path,
        [string[]]$Columns
    )

    $result = @{}
    foreach ($column in $Columns) {
        $result[$column] = @{}
    }

    foreach ($row in Get-XlsxRows -Path $Path -SelectColumns $Columns) {
        foreach ($column in $Columns) {
            $rawValue = [string]$row.$column
            $normValue = Normalize-Id -Value $rawValue
            if ([string]::IsNullOrWhiteSpace($normValue)) {
                continue
            }

            if (-not $result[$column].ContainsKey($normValue)) {
                $result[$column][$normValue] = $rawValue.Trim()
            }
        }
    }

    return $result
}

Write-Host "Step 1/6: 鏋勫缓 student_master 鍩虹琛?.."

$studentMasterMap = @{}

foreach ($row in Get-XlsxRows -Path ".\瀛︾敓鍩烘湰淇℃伅.xlsx" -SelectColumns @("XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "XB" -Value $row.XB
    Set-FirstNonEmpty -Record $record -Field "MZMC" -Value $row.MZMC
    Set-FirstNonEmpty -Record $record -Field "ZZMMMC" -Value $row.ZZMMMC
    Set-FirstNonEmpty -Record $record -Field "CSRQ" -Value $row.CSRQ
    Set-FirstNonEmpty -Record $record -Field "JG" -Value $row.JG
    Set-FirstNonEmpty -Record $record -Field "XSM" -Value $row.XSM
    Set-FirstNonEmpty -Record $record -Field "ZYM" -Value $row.ZYM
    Add-RecordSource -Record $record -SourceName "瀛︾敓鍩烘湰淇℃伅.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\瀛︾敓浣撹兘鑰冩牳.xlsx" -SelectColumns @("XM", "XH", "XB", "NJ", "YX", "ZY", "BJ", "BH")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "XM" -Value $row.XM
    Set-FirstNonEmpty -Record $record -Field "XB" -Value $row.XB
    Set-FirstNonEmpty -Record $record -Field "NJ" -Value $row.NJ
    Set-FirstNonEmpty -Record $record -Field "YX" -Value $row.YX
    Set-FirstNonEmpty -Record $record -Field "ZY" -Value $row.ZY
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJ
    Set-FirstNonEmpty -Record $record -Field "BH" -Value $row.BH
    Add-RecordSource -Record $record -SourceName "瀛︾敓浣撹兘鑰冩牳.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\浣撴祴鏁版嵁.xlsx" -SelectColumns @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "XB" -Value $row.XB
    Set-FirstNonEmpty -Record $record -Field "NJ" -Value $row.NJ
    Set-FirstNonEmpty -Record $record -Field "YX" -Value $row.YX
    Set-FirstNonEmpty -Record $record -Field "ZY" -Value $row.ZY
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJ
    Set-FirstNonEmpty -Record $record -Field "BH" -Value $row.BH
    Add-RecordSource -Record $record -SourceName "浣撴祴鏁版嵁.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\浣撹偛璇?xlsx" -SelectColumns @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "XB" -Value $row.XB
    Set-FirstNonEmpty -Record $record -Field "NJ" -Value $row.NJ
    Set-FirstNonEmpty -Record $record -Field "YX" -Value $row.YX
    Set-FirstNonEmpty -Record $record -Field "ZY" -Value $row.ZY
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJ
    Set-FirstNonEmpty -Record $record -Field "BH" -Value $row.BH
    Add-RecordSource -Record $record -SourceName "浣撹偛璇?xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\鏃ュ父閿荤偧.xlsx" -SelectColumns @("XH", "XB", "NJ", "YX", "ZY", "BJ", "BH")) {
    $studentId = Normalize-Id -Value $row.XH
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "XB" -Value $row.XB
    Set-FirstNonEmpty -Record $record -Field "NJ" -Value $row.NJ
    Set-FirstNonEmpty -Record $record -Field "YX" -Value $row.YX
    Set-FirstNonEmpty -Record $record -Field "ZY" -Value $row.ZY
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJ
    Set-FirstNonEmpty -Record $record -Field "BH" -Value $row.BH
    Add-RecordSource -Record $record -SourceName "鏃ュ父閿荤偧.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\姣曚笟鍘诲悜.xlsx" -SelectColumns @("SID", "GRADE", "BJMC", "BYQXMC")) {
    $studentId = Normalize-Id -Value $row.SID
    if ([string]::IsNullOrWhiteSpace($studentId)) {
        continue
    }

    $record = Get-OrCreateStudentRecord -StudentMap $studentMasterMap -StudentId $studentId
    Set-FirstNonEmpty -Record $record -Field "GRADE" -Value $row.GRADE
    Set-FirstNonEmpty -Record $record -Field "BJ" -Value $row.BJMC
    Set-FirstNonEmpty -Record $record -Field "BYQXMC" -Value $row.BYQXMC
    Add-RecordSource -Record $record -SourceName "姣曚笟鍘诲悜.xlsx"
}

$studentMasterRows = [System.Collections.Generic.List[object]]::new()
$orgClassIndex = @{}
$orgMajorIndex = @{}

foreach ($studentId in ($studentMasterMap.Keys | Sort-Object)) {
    $record = $studentMasterMap[$studentId]

    if ([string]::IsNullOrWhiteSpace($record.YX)) {
        $record.YX = $record.XSM
    }
    if ([string]::IsNullOrWhiteSpace($record.ZY)) {
        $record.ZY = $record.ZYM
    }
    if ([string]::IsNullOrWhiteSpace($record.BJ)) {
        $record.BJ = $record.BH
    }

    $departmentNorm = Normalize-Text -Value $record.YX
    $majorNorm = Normalize-Text -Value $record.ZY
    $classNorm = Normalize-Text -Value $record.BJ
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
        department_norm = $departmentNorm
        major_norm      = $majorNorm
        class_norm      = $classNorm
        org_class_key   = $orgClassKey
        org_major_key   = $orgMajorKey
        source_tables   = Join-Set -Set $record.source_tables
    }) | Out-Null
}

Write-Host "Step 2/6: 鏋勫缓 account_master 骞舵敞鍐岃处鍙锋潵婧?.."

$accountMasterMap = @{}
$accountSourceValues = @{}

foreach ($row in Get-XlsxRows -Path ".\绾夸笂瀛︿範锛堢患鍚堣〃鐜帮級.xlsx" -SelectColumns @("LOGIN_NAME", "XM", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME", "ROLEID")) {
    if (($row.ROLEID).Trim() -ne "3") {
        continue
    }

    $loginRaw = [string]$row.LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) {
        continue
    }

    Register-SourceValue -ValueStore $accountSourceValues -SourceTable "绾夸笂瀛︿範锛堢患鍚堣〃鐜帮級.xlsx" -SourceField "LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "user_name" -Value $row.XM
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.DEPARTMENT_NAME
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJOR_NAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASS_NAME
    Add-RecordSource -Record $record -SourceName "绾夸笂瀛︿範锛堢患鍚堣〃鐜帮級.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\璇惧爞浠诲姟鍙備笌.xlsx" -SelectColumns @("LOGIN_NAME", "USER_NAME", "DEPARTMENT_NAME", "MAJOR_NAME", "CLASS_NAME")) {
    $loginRaw = [string]$row.LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) {
        continue
    }

    Register-SourceValue -ValueStore $accountSourceValues -SourceTable "璇惧爞浠诲姟鍙備笌.xlsx" -SourceField "LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "user_name" -Value $row.USER_NAME
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.DEPARTMENT_NAME
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJOR_NAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASS_NAME
    Add-RecordSource -Record $record -SourceName "璇惧爞浠诲姟鍙備笌.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\瀛︾敓绛惧埌璁板綍.xlsx" -SelectColumns @("LOGIN_NAME", "XYMC", "MAJORNAME", "CLASSNAME", "ROLE")) {
    if (($row.ROLE).Trim() -ne "3") {
        continue
    }

    $loginRaw = [string]$row.LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) {
        continue
    }

    Register-SourceValue -ValueStore $accountSourceValues -SourceTable "瀛︾敓绛惧埌璁板綍.xlsx" -SourceField "LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.XYMC
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJORNAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASSNAME
    Add-RecordSource -Record $record -SourceName "瀛︾敓绛惧埌璁板綍.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\瀛︾敓浣滀笟鎻愪氦璁板綍.xlsx" -SelectColumns @("CREATER_LOGIN_NAME", "CREATER_NAME", "COLLEDGE_NAME", "MAJOR_NAME", "CLASS_NAME")) {
    $loginRaw = [string]$row.CREATER_LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) {
        continue
    }

    Register-SourceValue -ValueStore $accountSourceValues -SourceTable "瀛︾敓浣滀笟鎻愪氦璁板綍.xlsx" -SourceField "CREATER_LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "user_name" -Value $row.CREATER_NAME
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.COLLEDGE_NAME
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJOR_NAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASS_NAME
    Add-RecordSource -Record $record -SourceName "瀛︾敓浣滀笟鎻愪氦璁板綍.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\鑰冭瘯鎻愪氦璁板綍.xlsx" -SelectColumns @("CREATER_LOGIN_NAME", "CREATER_NAME", "COLLEDGE_NAME", "MAJOR_NAME", "CLASS_NAME")) {
    $loginRaw = [string]$row.CREATER_LOGIN_NAME
    $loginNorm = Normalize-Id -Value $loginRaw
    if ([string]::IsNullOrWhiteSpace($loginNorm)) {
        continue
    }

    Register-SourceValue -ValueStore $accountSourceValues -SourceTable "鑰冭瘯鎻愪氦璁板綍.xlsx" -SourceField "CREATER_LOGIN_NAME" -RawValue $loginRaw
    $record = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $loginNorm -LoginRaw $loginRaw.Trim()
    Set-FirstNonEmpty -Record $record -Field "user_name" -Value $row.CREATER_NAME
    Set-FirstNonEmpty -Record $record -Field "department_name" -Value $row.COLLEDGE_NAME
    Set-FirstNonEmpty -Record $record -Field "major_name" -Value $row.MAJOR_NAME
    Set-FirstNonEmpty -Record $record -Field "class_name" -Value $row.CLASS_NAME
    Add-RecordSource -Record $record -SourceName "鑰冭瘯鎻愪氦璁板綍.xlsx"
}

foreach ($row in Get-XlsxRows -Path ".\璁ㄨ璁板綍.xlsx" -SelectColumns @("CREATER_LOGIN_NAME", "CREATER_NAME", "CREATER_ROLE", "REPLY_LOGIN_NAME", "REPLY_USER_NAME", "REPLY_USER_ROLE")) {
    if (($row.CREATER_ROLE).Trim() -eq "3") {
        $creatorLoginRaw = [string]$row.CREATER_LOGIN_NAME
        $creatorLoginNorm = Normalize-Id -Value $creatorLoginRaw
        if (-not [string]::IsNullOrWhiteSpace($creatorLoginNorm)) {
            Register-SourceValue -ValueStore $accountSourceValues -SourceTable "璁ㄨ璁板綍.xlsx" -SourceField "CREATER_LOGIN_NAME" -RawValue $creatorLoginRaw
            $creatorRecord = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $creatorLoginNorm -LoginRaw $creatorLoginRaw.Trim()
            Set-FirstNonEmpty -Record $creatorRecord -Field "user_name" -Value $row.CREATER_NAME
            Add-RecordSource -Record $creatorRecord -SourceName "璁ㄨ璁板綍.xlsx"
        }
    }

    if (($row.REPLY_USER_ROLE).Trim() -eq "3") {
        $replyLoginRaw = [string]$row.REPLY_LOGIN_NAME
        $replyLoginNorm = Normalize-Id -Value $replyLoginRaw
        if (-not [string]::IsNullOrWhiteSpace($replyLoginNorm)) {
            Register-SourceValue -ValueStore $accountSourceValues -SourceTable "璁ㄨ璁板綍.xlsx" -SourceField "REPLY_LOGIN_NAME" -RawValue $replyLoginRaw
            $replyRecord = Get-OrCreateAccountRecord -AccountMap $accountMasterMap -LoginNorm $replyLoginNorm -LoginRaw $replyLoginRaw.Trim()
            Set-FirstNonEmpty -Record $replyRecord -Field "user_name" -Value $row.REPLY_USER_NAME
            Add-RecordSource -Record $replyRecord -SourceName "璁ㄨ璁板綍.xlsx"
        }
    }
}

Write-Host "Step 3/6: 瑙ｆ瀽璐﹀彿鍒?student_id 鐨勫彲鐢ㄦ槧灏?.."

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

Write-Host "Step 4/6: 閲囬泦鐩存帴涓婚敭鏉ユ簮骞剁敓鎴?bridge 琛?.."

$bridgeRows = [System.Collections.Generic.List[object]]::new()

$directMappings = @(
    @{ Path = ".\瀛︾敓鍩烘湰淇℃伅.xlsx";       SourceTable = "瀛︾敓鍩烘湰淇℃伅.xlsx";       Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\瀛︾敓鎴愮哗.xlsx";           SourceTable = "瀛︾敓鎴愮哗.xlsx";           Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\瀛︾敓閫夎淇℃伅.xlsx";       SourceTable = "瀛︾敓閫夎淇℃伅.xlsx";       Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\瀛︾睄寮傚姩.xlsx";           SourceTable = "瀛︾睄寮傚姩.xlsx";           Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\鏃ュ父閿荤偧.xlsx";           SourceTable = "鏃ュ父閿荤偧.xlsx";           Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\浣撹偛璇?xlsx";             SourceTable = "浣撹偛璇?xlsx";             Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\浣撴祴鏁版嵁.xlsx";           SourceTable = "浣撴祴鏁版嵁.xlsx";           Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\瀛︾敓浣撹兘鑰冩牳.xlsx";       SourceTable = "瀛︾敓浣撹兘鑰冩牳.xlsx";       Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\鑰冨嫟姹囨€?xlsx";           SourceTable = "鑰冨嫟姹囨€?xlsx";           Field = "XH";       MatchType = "exact";          MatchConfidence = "high"; Evidence = "XH" },
    @{ Path = ".\涓婄綉缁熻.xlsx";           SourceTable = "涓婄綉缁熻.xlsx";           Field = "XSBH";     MatchType = "exact";          MatchConfidence = "high"; Evidence = "XSBH" },
    @{ Path = ".\濂栧閲戣幏濂?xlsx";         SourceTable = "濂栧閲戣幏濂?xlsx";         Field = "XSBH";     MatchType = "exact";          MatchConfidence = "high"; Evidence = "XSBH" },
    @{ Path = ".\绀惧洟娲诲姩.xlsx";           SourceTable = "绀惧洟娲诲姩.xlsx";           Field = "XSBH";     MatchType = "exact";          MatchConfidence = "high"; Evidence = "XSBH" },
    @{ Path = ".\姣曚笟鍘诲悜.xlsx";           SourceTable = "姣曚笟鍘诲悜.xlsx";           Field = "SID";      MatchType = "exact";          MatchConfidence = "high"; Evidence = "SID" },
    @{ Path = ".\璺戞鎵撳崱.xlsx";           SourceTable = "璺戞鎵撳崱.xlsx";           Field = "USERNUM";  MatchType = "exact";          MatchConfidence = "high"; Evidence = "USERNUM" },
    @{ Path = ".\闂ㄧ鏁版嵁.xlsx";           SourceTable = "闂ㄧ鏁版嵁.xlsx";           Field = "IDSERTAL"; MatchType = "exact";          MatchConfidence = "high"; Evidence = "IDSERTAL" },
    @{ Path = ".\鍥句功棣嗘墦鍗¤褰?xlsx";     SourceTable = "鍥句功棣嗘墦鍗¤褰?xlsx";     Field = "cardld";   MatchType = "exact";          MatchConfidence = "high"; Evidence = "cardld" },
    @{ Path = ".\鍥涘叚绾ф垚缁?xlsx";         SourceTable = "鍥涘叚绾ф垚缁?xlsx";         Field = "KS_XH";    MatchType = "exact";          MatchConfidence = "high"; Evidence = "KS_XH" },
    @{ Path = ".\瀛︾绔炶禌.xlsx";           SourceTable = "瀛︾绔炶禌.xlsx";           Field = "XHHGH";    MatchType = "filtered_exact"; MatchConfidence = "high"; Evidence = "XHHGH filtered by 瀛︾敓鍩烘湰淇℃伅.XH" }
)

foreach ($mapping in $directMappings) {
    Write-Host ("  - 鎵弿 {0}::{1}" -f $mapping.SourceTable, $mapping.Field)
    $uniqueValues = (Get-UniqueValueMap -Path $mapping.Path -Columns @($mapping.Field))[$mapping.Field]
    foreach ($valueNorm in ($uniqueValues.Keys | Sort-Object)) {
        $valueRaw = $uniqueValues[$valueNorm]
        $studentId = ""
        $matchType = $mapping.MatchType
        $matchConfidence = $mapping.MatchConfidence

        if ($studentMasterMap.ContainsKey($valueNorm)) {
            $studentId = $valueNorm
        }
        else {
            if ($mapping.Field -eq "XHHGH") {
                $matchType = "filtered_out_nonstudent"
                $matchConfidence = "none"
            }
            else {
                $matchType = "orphan_exact"
                $matchConfidence = "none"
            }
        }

        Add-BridgeRow -BridgeRows $bridgeRows `
            -SourceTable $mapping.SourceTable `
            -SourceField $mapping.Field `
            -SourceValueRaw $valueRaw `
            -SourceValueNorm $valueNorm `
            -StudentId $studentId `
            -MatchType $matchType `
            -MatchConfidence $matchConfidence `
            -EvidenceFields $mapping.Evidence
    }
}

Write-Host "Step 5/6: 杩藉姞璐﹀彿绫?bridge 琛?.."

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

        Add-BridgeRow -BridgeRows $bridgeRows `
            -SourceTable $sourceTable `
            -SourceField $sourceField `
            -SourceValueRaw $valueRaw `
            -SourceValueNorm $valueNorm `
            -StudentId $studentId `
            -MatchType $matchType `
            -MatchConfidence $matchConfidence `
            -EvidenceFields $evidenceFields
    }
}

Write-Host "Step 6/6: 瀵煎嚭缁撴灉鏂囦欢..."

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
    "- direct id fields are written as exact or filtered_exact matches",
    "- account fields are first aggregated into account_master, then mapped only when the organization tuple is uniquely identifiable",
    "- unresolved account rows are kept unresolved instead of being force-matched",
    "- running data is included in the bridge scan, but the Excel row-limit risk should still be checked before modeling",
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
