echo "- Downloading enwik8 (Character)"
mkdir -p enwik8
cd enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
cd ..
python tokenize_enwiki.py
echo "---"
echo "- Done!"