-- Export Apple Notes to dated Markdown files for `morpheus import-journal`.
--
-- Apple Notes has no usable bulk export: "Export as PDF" destroys the text and
-- copy-paste loses the dates. This walks a folder and writes one file per note,
-- named by the note's creation date, which is what the importer reads.
--
-- Usage:
--   1. Edit FOLDER_NAME below to match your notes folder ("Notes" is the default
--      folder; use whatever yours is actually called).
--   2. osascript scripts/export-apple-notes.applescript
--   3. Files land in ~/morpheus-journal-export/
--
-- macOS will ask for permission to control Notes the first time. That prompt is
-- the system asking on your behalf; nothing leaves your machine.

set FOLDER_NAME to "Notes"
set OUTPUT_DIR to (POSIX path of (path to home folder)) & "morpheus-journal-export/"

do shell script "mkdir -p " & quoted form of OUTPUT_DIR

set exported to 0
set skipped to 0

tell application "Notes"
	set targetFolder to missing value
	repeat with f in folders
		if name of f is FOLDER_NAME then set targetFolder to f
	end repeat

	if targetFolder is missing value then
		return "No folder named \"" & FOLDER_NAME & "\". Folders available: " & ¬
			(name of folders as string)
	end if

	repeat with n in notes of targetFolder
		set noteBody to plaintext of n
		set noteDate to creation date of n

		-- ISO date for the filename; the importer parses it from the stem.
		set y to year of noteDate as integer
		set m to (month of noteDate as integer)
		set d to day of noteDate
		set mm to text -2 thru -1 of ("0" & m)
		set dd to text -2 thru -1 of ("0" & d)
		set stamp to (y as string) & "-" & mm & "-" & dd

		-- Several notes can share a date; suffix so none overwrite another.
		set suffix to 0
		set filePath to OUTPUT_DIR & stamp & ".md"
		repeat while (do shell script "test -e " & quoted form of filePath & ¬
			" && echo yes || echo no") is "yes"
			set suffix to suffix + 1
			set filePath to OUTPUT_DIR & stamp & "-" & (suffix as string) & ".md"
		end repeat

		if length of noteBody > 0 then
			do shell script "cat > " & quoted form of filePath & " <<'MORPHEUSEOF'
" & noteBody & "
MORPHEUSEOF"
			set exported to exported + 1
		else
			set skipped to skipped + 1
		end if
	end repeat
end tell

return "exported " & exported & " notes to " & OUTPUT_DIR & ¬
	" (skipped " & skipped & " empty)"
