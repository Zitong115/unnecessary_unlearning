conda activate py38_torch1.8.1_cu111_zitong

cd /home/zitong/necessary_unlearning/src

python main.py --config ../config/Distribution.yml # first generate distribution statistic for MNIST, CIFAR-10, CIFAR-100

python main.py --config ../config/filter_rfmodel_MNIST_0.1.yml

python main.py --config ../config/filter_rfmodel_MNIST_0.01.yml

python main.py --config ../config/filter_rfmodel_MNIST_0.05.yml

python main.py --config ../config/filter_rfmodel_MNIST_class.yml

python main.py --config ../config/filter_rfmodel_MNIST_class.yml

-

python main.py --config ../config/SISA.yml


conda activate py38_torch1.8.1_cu111_zitong
cd /home/zitong/necessary_unlearning/src
python main.py --config ../config/filter_rfmodel_CIFAR10_0.1.yml

conda activate py38_torch1.8.1_cu111_zitong
cd /home/zitong/necessary_unlearning/src
nohup python main.py --config ../config/filter_rfmodel_CIFAR10_0.01.yml


conda activate py38_torch1.8.1_cu111_zitong
cd /home/zitong/necessary_unlearning/src
nohup python main.py --config ../config/filter_rfmodel_CIFAR10_0.05.yml

conda activate py38_torch1.8.1_cu111_zitong
cd /home/zitong/necessary_unlearning/src
nohup python main.py --config ../config/filter_rfmodel_CIFAR10_class_0.5.yml

conda activate py38_torch1.8.1_cu111_zitong
cd /home/zitong/necessary_unlearning/src
nohup python main.py --config ../config/filter_rfmodel_CIFAR100_30.yml > cifar100_30_set1.out

nohup python main.py --config ../config/filter_rfmodel_CIFAR10_class_0.25.yml > cifar10_0.25class_set1.out
nohup python main.py --config ../config/filter_rfmodel_CIFAR10_class_0.25.yml > cifar10_0.25class_set3.out
nohup python main.py --config ../config/filter_rfmodel_CIFAR100_class_0.9.yml > cifar100_0.9class_set3.out
/home/zitong/necessary_unlearning/config/filter_rfmodel_CIFAR100_class_0.5.yml
-

python main.py --config ../config/CIFAR100-ResNet-18.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_0.1.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_0.01.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_0.05.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_class_0.5.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_class_0.05.yml

-

python main.py --config ../config/filter_rfmodel_CIFAR100_10.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_30.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_50.yml

python main.py --config ../config/filter_rfmodel_CIFAR100_100.yml

python main.py --config ../config/filter_rfmodel_MNIST_100.yml