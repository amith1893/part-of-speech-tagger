awk '{print $2}' $1 > word_per_line.txt
awk '{print $3}' $1 > tag_per_line.txt

## Remove the newline characters

awk 'NF' word_per_line.txt > word.txt
awk 'NF' tag_per_line.txt > tag.txt

tail --lines=+2 word.txt | paste word.txt - | sort | uniq -c | sort -rn > word_bigram.txt
tail --lines=+2 tag.txt | paste tag.txt - | sort | uniq -c | sort -rn > tag_bigram.txt

cat tag.txt | sort | uniq -c | sort -rn >  tag_unigram.txt
cat word.txt | sort | uniq -c | sort -rn >  word_unigram.txt

##cleaning

rm word_per_line.txt tag_per_line.txt word.txt tag.txt


