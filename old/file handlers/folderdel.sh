for FOLDER in $(find ./typingrecords/ -mindepth 1 -maxdepth 1 -type d); do
    rm -r "$FOLDER"
    echo "removed folder $FOLDER"
done