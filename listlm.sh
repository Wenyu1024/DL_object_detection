#
#	lIFE iS sHORT tO lABEL mANUALLY
#		   listlm
#
#	    Alper ALTINOK, 2017
#	
#User variables, change these to your needs.
TESTIMAGES=/wenyu/20180128-182941-51a2_epoch_79.0/*.png # Provide path and filetype info(*.png or *.jpg or *.gif, etc..) here
OUTPUTFILES=/wenyu/20180128-182941-51a2_epoch_79.0/*.txt # Provide a path for label ".txt" files
JOBNAME="20180128-182941-51a2" #Change the Job-ID here. Provide a name from jobs directory.
LABELNAME=cell    # Choose a Detectnet label,  car  is default. Only one label is possible. You can create a custom label.
#------end of user definitions-----------------

for fn in $TESTIMAGES
do
bname=$(basename "$fn")
bnamext=$(basename "$fn" | cut -d. -f1)
pathname=$(dirname "$OUTPUTFILES")

curl http://localhost:5000/models/images/generic/infer_one.json -XPOST -F job_id=$JOBNAME -F dont_resize='dont_resize' -F snapshot_epoch=29 -F image_file=@$fn > "$pathname"/"$bnamext.txt"
done
for f in $OUTPUTFILES
do
	echo "Current file is :  $f"

sed -i '$d ; 3s/    "bbox-list": // ; 1,2d' ${f}
sed -i '$d' ${f}
python2 json2csv.py $f ${f}
sed -i '1d' ${f}
sed -ri 's/([ \,]+[^ \,]*){1}$//' ${f}
sed -ri 's/([^,]*,[^,]*,[^,]*,[^,]*),[^,]*,/\1\n/g' ${f}
sed -i '/0.0,0.0,0.0,0.0/d' ${f}
sed -i 's/.*/'"$LABELNAME"',0.0,0.0,0.0,&,0.0,0.0,0.0,0.0,0.0,0.0,0.0/' ${f}
sed -i 's/,/ /g' ${f}
done
