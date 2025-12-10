FILE="pyproject.toml" # Replace with your actual file name
REGEX='version = "0\.2\.\([0-9]\+\)"'

# Extract the current patch number
current_N=$(sed -nE "s/${REGEX}/\\1/p" "${FILE}")

if [[ -z "$current_N" ]]; then
    echo "Could not find the version number in ${FILE}"
    exit 1
fi

# Calculate the new patch number
new_N=$((current_N + 1))

# Replace the line in the file
# Note: macOS sed requires an empty string backup suffix (the '' part)
sed -i '' -E "s/${REGEX}/version = \"0.2.${new_N}\"/" "${FILE}"

echo "Version incremented to 0.2.${new_N} in ${FILE}"
nano src/code_similarity_engine/__init__.py
rm -rf dist
python -m build 
twine upload dist/*
