#!/bin/sh

i=1
counter=0
while [ $i -lt 2 ]; do
	echo File downloaded!
	wget --header "Cookie: PHPSESSID=h97fb19ufkl7s4oalf5rdmdtul" -nv -q http://vision.ctf/catchat.php -O /root/Documents/nsec19/calibrate/tensorflow-image-detection/image.png
	python3 /root/Documents/nsec19/calibrate/tensorflow-image-detection/classify.py /root/Documents/nsec19/calibrate/tensorflow-image-detection/image.png
	let counter=counter+1
	echo Try: $counter
	echo ""
done

# download the image
#wget httYp://vision.ctf/catchat.php -O image.png






