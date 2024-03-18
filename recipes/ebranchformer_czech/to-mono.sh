for i in `find ./2020* -type f` ; do c=`soxi -c $i`; if (($c > 1)) ;then echo $i; sox $i -t wav -c 1 ${i}_1 ; mv ${i}_1 ${i} ; fi ; done
