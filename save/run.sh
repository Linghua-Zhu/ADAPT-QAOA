for batch in {0..30}
do
	for value in {1..3}
	do
		sample=$(($value+$(($batch*3)) ))
		echo $sample
       		python hdvv_1d_vqe.py -K=-0.5 -s=$sample >> jk_1-05.$sample.txt &
 	     	python hdvv_1d_vqe.py -K=-1.0 -s=$sample >> jk_1-10.$sample.txt &
               	python hdvv_1d_vqe.py -K=-1.5 -s=$sample >> jk_1-15.$sample.txt &
               	python hdvv_1d_vqe.py -K=-2.0 -s=$sample >> jk_1-20.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-2.5 -s=$sample >> jk_1-25.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-3.0 -s=$sample >> jk_1-30.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-3.5 -s=$sample >> jk_1-35.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-4.0 -s=$sample >> jk_1-40.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-4.5 -s=$sample >> jk_1-45.$sample.txt &
 	      	python hdvv_1d_vqe.py -K=-5.0 -s=$sample >> jk_1-50.$sample.txt &
	done

	wait
done
