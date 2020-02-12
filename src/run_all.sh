#! /bin/zsh

time_stamp=`date +"%Y%m%d-%H%M%S"`;
file_name="result-${time_stamp}.zip"

#shasum report01.py > $file_name
#poetry run python3 report01.py >> $file_name
#shasum ../img/p1.png >> $file_name
#echo "" >> $file_name

shasum report02.py > $file_name
poetry run python3 report02.py >> $file_name
shasum ../img/p2.png >> $file_name
echo "" >> $file_name

shasum report03.py >> $file_name
poetry run python3 report03.py >> $file_name
shasum ../img/p3.png >> $file_name
echo "" >> $file_name

shasum report04.py >> $file_name
poetry run python3 report04.py >> $file_name
shasum ../img/p4.png >> $file_name
echo "" >> $file_name

shasum report05_m2.py >> $file_name
poetry run python3 report05_m2.py >> $file_name
shasum ../img/p5_m2.png >> $file_name
echo "" >> $file_name

shasum report05_m3.py >> $file_name
poetry run python3 report05_m3.py >> $file_name
shasum ../img/p5_m3.png >> $file_name
echo "" >> $file_name

shasum report05_m5.py >> $file_name
poetry run python3 report05_m5.py >> $file_name
shasum ../img/p5_m5.png >> $file_name
echo "" >> $file_name

cat $file_name