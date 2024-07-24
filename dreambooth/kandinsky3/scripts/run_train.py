import os

def main():
    os.system("sbatch train.sh dog6 dog sks -C \"type_e\"")
    # os.system("sbatch train.sh berry_bowl bowl sks")
    # os.system("sbatch train.sh clock clock sks")
    # os.system("sbatch train.sh backpack_dog backpack sks")
    # os.system("sbatch train.sh can can sks")
    # os.system("sbatch train.sh cat cat sks")
    # os.system("sbatch train.sh monster_toy toy sks")

if __name__ == '__main__':
    main()
