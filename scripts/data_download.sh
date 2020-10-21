#CIFAR10, CIFAR100
# python3 datautil/cifar.py

# HPatches
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -O data/hpatches-sequences-release
tar -xvzf data/hpatches-sequences-release.tar.gz -C data
cd data/hpatches-sequences-release
rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
cd ../..