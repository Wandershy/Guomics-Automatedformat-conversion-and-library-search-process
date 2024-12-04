import os
import argparse
import pandas as pd
import numpy as np
import re
import subprocess
import time
import multiprocessing
import threading
from functools import partial

parser = argparse.ArgumentParser(description="pipeline for metabolism cross analysis")
parser.add_argument("-p","--speciesPath",dest="speciesPath",required=True,type=str,help="species_Path")
parser.add_argument("-i","--input",dest="input",required=True,type=str,help="input species list")
parse = parser.parse_args()

def generateDir(speciesPath_dir):
    ori_total_dir = "/storage/guotiannanLab/xingziyuan/01.metaMS-GPT/00.training_dataset/00.workflow.temp"
    total_dir = speciesPath_dir
    raw_dir = os.path.join(total_dir, '00.rawdata')
    workflow_dir = os.path.join(total_dir, '01.workflow')
    fasta_dir = os.path.join(total_dir, '02.fasta')
    FP_dir = os.path.join(total_dir, '03.FragPipe')
    sage_dir = os.path.join(total_dir, '04.Sage')
    res_dir = os.path.join(total_dir, '05.All_results')

    try:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(workflow_dir, exist_ok=True)
        os.makedirs(fasta_dir, exist_ok=True)
        os.makedirs(FP_dir, exist_ok=True)
        os.makedirs(sage_dir, exist_ok=True)
        os.makedirs(os.path.join(res_dir, "00.FragPipe"), exist_ok=True)
        os.makedirs(os.path.join(res_dir, "01.Sage"), exist_ok=True)
    except Exception as e:
        print(f"创建目录时发生错误：{e}")

    command1 = "cp -r " + ori_total_dir + "/01.workflow/00.FP_workflow_ori " + ori_total_dir + "/01.workflow/01.Sage_workflow_ori " + ori_total_dir + "/01.workflow/msConvert.config.txt " + workflow_dir
    try:
        os.system(command1)
    except:
        raise SystemExit
    ## temp dir
    os.makedirs(os.path.join(total_dir, 'tmp'), exist_ok=True)

def generate_decoy_fasta(speciesPath_dir):
    sh_file_path = os.path.join(speciesPath_dir, "00.generate.FP.decoy.fasta.sh")
    lines = [
        "#!/bin/bash\n",
        "#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short\n",
        "#SBATCH -q normal\n",
        "#SBATCH -J MS-GPT_decoy\n",
        "#SBATCH -c 5\n",
        "#SBATCH --mem 10G\n",
        "################# set path ##########################\n",
        "ori_total_dir=/storage/guotiannanLab/xingziyuan/01.metaMS-GPT/00.training_dataset/00.workflow.temp\n",
        "total_dir=" + speciesPath_dir + "\n",
        "raw_dir=$total_dir/00.rawdata\n",
        "workflow_dir=$total_dir/01.workflow\n",
        "fasta_dir=$total_dir/02.fasta\n",
        "FP_dir=$total_dir/03.FragPipe\n",
        "sage_dir=$total_dir/04.Sage\n",
        "res_dir=$total_dir/05.All_results\n",
        "msConvert_workflow=$workflow_dir/msConvert.config.txt\n",
        "FP_workflow=$workflow_dir/00.FP_workflow_ori/fasta_LFQ_MBR_DDA_target_precolator_revise_modi.workflow\n",
        "sage_workflow=$workflow_dir/sage.config.json\n",
        "mydate=`date`\n",
        "################### FragPipe 21.1 generate decoy fasta #####################\n",
        "# change temp folder\n",
        "export JAVA_OPTS=-Djava.io.tmpdir=$total_dir/tmp\n",
        "echo $JAVA_OPTS\n",
        "XDG_CONFIG_HOME=$total_dir\n",
        "export XDG_CONFIG_HOME\n",
        "# add decoys to fasta\n",
        "## software dir and para\n",
        "fragpipePathDir=\"/home/guotiannanLab/xingziyuan/software/fragpipe21\"\n",
        "philosopherPath=\"$fragpipePathDir/software/philosopher-5.1.0/philosopher\"\n",
        "cd $fasta_dir\n",
        "$philosopherPath workspace --clean --nocheck\n",
        "$philosopherPath workspace --init --nocheck\n",
        "$philosopherPath database --custom $fasta_dir/*.fasta\n",
        "$philosopherPath workspace --clean --nocheck\n",
        "decoyfasta=`ls $fasta_dir/*.fasta.fas`\n",
        "echo $decoyfasta\n",
        "cp $FP_workflow $workflow_dir/fasta_LFQ_MBR_DDA_target_precolator_revise_modi.workflow\n",
        "echo \"database.db-path=${decoyfasta}\" >> $workflow_dir/fasta_LFQ_MBR_DDA_target_precolator_revise_modi.workflow\n",
    ]
    try:
        with open(sh_file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"00.generate.FP.decoy.fasta.sh 已成功写入到: {sh_file_path}")
    except Exception as e:
        print(f"写入脚本时发生错误: {e}")
    pass

def msConvert(speciesPath_dir, raw_dir, raw_file_count):
    sh_file_path = os.path.join(speciesPath_dir, "01.msConvert_rerun.sh")
    lines = [
        "#!/bin/bash\n",
        "#slurm options\n",
        "#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short\n",
        "#SBATCH -q huge\n",
        "#SBATCH -J MS-GPT_msConvert_array\n",
        "#SBATCH -c 5\n",
        "#SBATCH -a 1-" + str(raw_file_count) + "\n",
        "########################## MSConvert run #####################\n",
        "# module\n",
        "source ~/.bashrc\n",
        "module load singularity/3.7.1\n",
        "# dir\n",
        "total_dir=" + speciesPath_dir + "\n",
        "raw_dir=$total_dir/" + raw_dir + "\n",
        "workflow_dir=$total_dir/01.workflow\n",
        "temp_dir=$total_dir/tmp\n",
        "#mkdir $temp_dir\n",
        "# msconvert\n",
        "msconvert=/storage/guotiannanLab/xingziyuan/pwiz-skyline-i-agree-to-the-vendor-licenses_3.0.24054-2352758.sif\n",
        "# run\n",
        "ls $raw_dir/*.raw > rawdata.list\n",
        "id_list=rawdata.list\n",
        "id=`head -n $SLURM_ARRAY_TASK_ID $id_list | tail -n 1`\n",
        "filename_total=$id\n",
        "filename=$(basename $filename_total)\n",
        "echo $filename\n",
        "filenamepre=${filename%%.*}\n",
        "echo -e $filename_total > $temp_dir/${filenamepre}_msconvert_filelist.txt\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Start to run MSConvert: ${filename}\"\n",
        "#use config\n",
        "singularity exec -B /home -B /storage -B $temp_dir:/wineprefix64/drive_c/users/root/Temp $msconvert wine msconvert -f $temp_dir/${filenamepre}_msconvert_filelist.txt -o $raw_dir --inten64 -c $workflow_dir/msConvert.config.txt\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Finished MSConvert run: ${filename}\"\n",
    ]
    try:
        with open(sh_file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"01.msConvert_rerun.sh 已成功写入到: {sh_file_path}")
    except Exception as e:
        print(f"写入脚本时发生错误: {e}")
    pass

def FragPipe(speciesPath_dir, raw_dir, raw_file_count):
    sh_file_path = os.path.join(speciesPath_dir, "01.FragPipe.rerun.sh")
    lines = [
        "#!/bin/bash\n",
        "#slurm options\n",
        "#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short\n",
        "#SBATCH -q huge\n",
        "#SBATCH -J MS-GPT_FragPipe\n",
        "#SBATCH -c 10\n",
        "#SBATCH --mem 50G\n",
        "#SBATCH -a 1-" + str(raw_file_count) + "\n",
        "# dir\n",
        "total_dir=" + speciesPath_dir + "\n",
        "raw_dir=$total_dir/" + raw_dir + "\n",
        "workflow_dir=$total_dir/01.workflow\n",
        "fasta_dir=$total_dir/02.fasta\n",
        "temp_dir=$total_dir/tmp\n",
        "#mkdir $temp_dir\n",
        "########################## FragPipe run #####################\n",
        "#set path\n",
        "FP_work_dir=$total_dir/03.FragPipe\n",
        "# active python\n",
        "source /home/guotiannanLab/xingziyuan/anaconda3/bin/activate\n",
        "## software dir and para\n",
        "fragpipePathDir=\"/home/guotiannanLab/xingziyuan/software/fragpipe21\"\n",
        "fragpipePath=\"$fragpipePathDir/bin/fragpipe\"\n",
        "msfraggerPath=\"$fragpipePathDir/software/MSFragger-4.0/MSFragger-4.0.jar\"\n",
        "philosopherPath=\"$fragpipePathDir/software/philosopher-5.1.0/philosopher\"\n",
        "ionquantPath=\"$fragpipePathDir/software/IonQuant-1.10.12/IonQuant-1.10.12.jar\"\n",
        "pythonPath=/home/guotiannanLab/xingziyuan/anaconda3/bin/python\n",
        "workflowPath=$total_dir/01.workflow/fasta_LFQ_MBR_DDA_target_precolator_revise_modi.workflow\n",
        "## run\n",
        "ls $raw_dir/*.mzML > FP.mzML.list\n",
        "id_list=FP.mzML.list\n",
        "id=`head -n $SLURM_ARRAY_TASK_ID $id_list | tail -n 1`\n",
        "filename_total=$id\n",
        "filename=$(basename $filename_total)\n",
        "filenamepre=${filename%%.*}\n",
        "mkdir $FP_work_dir/$filenamepre\n",
        "echo -e ${filename_total}\"\\texp\\t\\tDDA\" > $FP_work_dir/$filenamepre/SampleList.manifest\n",
        "echo $filename_total\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Start to run FragPipe: ${filename}\"\n",
        "cd $FP_work_dir/$filenamepre\n",
        "$philosopherPath workspace --init --nocheck\n",
        "$philosopherPath workspace --clean --nocheck\n",
        "XDG_CONFIG_HOME=$FP_work_dir/$filenamepre\n",
        "export XDG_CONFIG_HOME\n",
        "cp $workflowPath $FP_work_dir/$filenamepre\n",
        "$fragpipePath --headless --workflow $FP_work_dir/$filenamepre/fasta_LFQ_MBR_DDA_target_precolator_revise_modi.workflow --manifest $FP_work_dir/$filenamepre/SampleList.manifest --workdir $FP_work_dir/$filenamepre --config-ionQuant $ionquantPath --config-msfragger $msfraggerPath --config-python $pythonPath --config-philosopher $philosopherPath --threads 20\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Finished FragPipe run: ${filename}\"\n",
        "cp $FP_work_dir/$filenamepre/exp/*_target_psms.tsv $total_dir/05.All_results/00.FragPipe\n",
    ]
    try:
        with open(sh_file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"01.FragPipe.rerun.sh 已成功写入到: {sh_file_path}")
    except Exception as e:
        print(f"写入脚本时发生错误: {e}")

def Sage(speciesPath_dir, raw_dir, raw_file_count):
    sh_file_path = os.path.join(speciesPath_dir, "01.Sage.rerun.sh")
    lines = [
        "#!/bin/bash\n",
        "#slurm options\n",
        "#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short\n",
        "#SBATCH -q huge\n",
        "#SBATCH -J MS-GPT_Sage\n",
        "#SBATCH -c 10\n",
        "#SBATCH --mem 10G\n",
        "#SBATCH -a 1-" + str(raw_file_count) + "\n",
        "# dir\n",
        "total_dir=" + speciesPath_dir + "\n",
        "raw_dir=$total_dir/" + raw_dir + "\n",
        "workflow_dir=$total_dir/01.workflow\n",
        "temp_dir=$total_dir/tmp\n",
        "#mkdir $temp_dir\n",
        "## run\n",
        "ls $raw_dir/*.mzML > Sage.mzML.list\n",
        "id_list=Sage.mzML.list\n",
        "id=`head -n $SLURM_ARRAY_TASK_ID $id_list | tail -n 1`\n",
        "filename_total=$id\n",
        "filename=$(basename $filename_total)\n",
        "filenamepre=${filename%%.*}\n",
        "#set path\n",
        "sage_work_dir=$total_dir/04.Sage\n",
        "sage_workflow_dir=$workflow_dir/01.Sage_workflow_ori\n",
        "fasta_dir=$total_dir/02.fasta\n",
        "fasta=`ls $fasta_dir/*.fasta`\n",
        "source activate sage\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Start to run Sage: ${filename}\"\n",
        "mkdir $sage_work_dir/$filenamepre\n",
        "sage -f $fasta -o  $sage_work_dir/$filenamepre $sage_workflow_dir/sage.config.json $filename_total\n",
        "mydate=`date`\n",
        "echo \"${mydate} ...\"\n",
        "echo \"Finished Sage run: ${filename}\"\n",
        "#################### cp to new dir ####################\n",
        "mv $sage_work_dir/$filenamepre/results.sage.tsv $sage_work_dir/$filenamepre/${filenamepre}_results.sage.tsv\n",
        "cp $sage_work_dir/$filenamepre/${filenamepre}_results.sage.tsv $total_dir/05.All_results/01.Sage\n",
    ]
    try:
        with open(sh_file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"01.Sage.rerun.sh 已成功写入到: {sh_file_path}")
    except Exception as e:
        print(f"写入脚本时发生错误: {e}")
        
def run_FragPipe(speciesPath_dir):
    # 7.FragPipe run
    # First run
    raw_dir = "00.rawdata"
    raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, raw_dir)) if f.endswith(".mzML")])
    FragPipe(speciesPath_dir, raw_dir, raw_count)
    # submit Slurm and acquire Job ID
    result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.FragPipe.rerun.sh")], capture_output=True, text=True)
    output = result.stdout.strip()
    job_id = output.split()[-1]  
    print(f"作业已提交，Job ID: {job_id}")
    while True:
        check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in check_result.stdout:
            print(f"作业 {job_id} 已完成或退出队列。")
            break
        else:
            print(f"作业 {job_id} 正在运行或等待中...")
        time.sleep(10)
        
    # Second run
    command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.FP.2nd"
    try:
        os.system(command)
    except:
        raise SystemExit
    command = "cd " + speciesPath_dir + "/00.rawdata\n" + "ls | grep .mzML > mzML.list\n" + "for mzML in `cat mzML.list`; do\nfilename=$(echo $mzML | sed 's/.mzML$//')\nif [ ! -f \"../05.All_results/00.FragPipe/${filename}_percolator_target_psms.tsv\" ]; then\ncp $mzML ../00.rawdata.FP.2nd; fi\ndone"
    try:
        os.system(command)
    except:
        raise SystemExit
    if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.FP.2nd')):
        raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.FP.2nd')) if f.endswith(".mzML")])
        raw_dir = '00.rawdata.FP.2nd'
        FragPipe(speciesPath_dir, raw_dir, raw_count)
        # submit Slurm and acquire Job ID
        result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.FragPipe.rerun.sh")], capture_output=True, text=True)
        output = result.stdout.strip()
        job_id = output.split()[-1]  
        print(f"作业已提交，Job ID: {job_id}")
        while True:
            check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id not in check_result.stdout:
                print(f"作业 {job_id} 已完成或退出队列。")
                break
            else:
                print(f"作业 {job_id} 正在运行或等待中...")
            time.sleep(10) 
        
        # Third run
        command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.FP.3rd"
        try:
            os.system(command)
        except:
            raise SystemExit
        command = "cd " + speciesPath_dir + "/00.rawdata\n" + "ls | grep .mzML > mzML.list\n" + "for mzML in `cat mzML.list`; do\nfilename=$(echo $mzML | sed 's/.mzML$//')\nif [ ! -f \"../05.All_results/00.FragPipe/${filename}_percolator_target_psms.tsv\" ]; then\ncp $mzML ../00.rawdata.FP.3rd; fi\ndone"
        try:
            os.system(command)
        except:
            raise SystemExit
        if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.FP.3rd')):
            raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.FP.3rd')) if f.endswith(".mzML")])
            raw_dir = '00.rawdata.FP.3rd'
            FragPipe(speciesPath_dir, raw_dir, raw_count)
            # submit Slurm and acquire Job ID
            result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.FragPipe.rerun.sh")], capture_output=True, text=True)
            output = result.stdout.strip()
            job_id = output.split()[-1]  
            print(f"作业已提交，Job ID: {job_id}")
            while True:
                check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
                if job_id not in check_result.stdout:
                    print(f"作业 {job_id} 已完成或退出队列。")
                    break
                else:
                    print(f"作业 {job_id} 正在运行或等待中...")
                time.sleep(10) 

def run_Sage(speciesPath_dir):
    # 8.Sage run
    # First run
    raw_dir = "00.rawdata"
    raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, raw_dir)) if f.endswith(".mzML")])
    Sage(speciesPath_dir, raw_dir, raw_count)
    # submit Slurm and acquire Job ID
    result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.Sage.rerun.sh")], capture_output=True, text=True)
    output = result.stdout.strip()
    job_id = output.split()[-1]  
    print(f"作业已提交，Job ID: {job_id}")
    while True:
        check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in check_result.stdout:
            print(f"作业 {job_id} 已完成或退出队列。")
            break
        else:
            print(f"作业 {job_id} 正在运行或等待中...")
        time.sleep(10)
        
    # Second run
    command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.Sage.2nd"
    try:
        os.system(command)
    except:
        raise SystemExit
    command = "cd " + speciesPath_dir + "/00.rawdata\n" + "ls | grep .mzML > mzML.list\n" + "for mzML in `cat mzML.list`; do\nfilename=$(echo $mzML | sed 's/.mzML$//')\nif [ ! -f \"../05.All_results/01.Sage/${filename}_results.sage.tsv\" ]; then\ncp $mzML ../00.rawdata.Sage.2nd; fi\ndone"
    try:
        os.system(command)
    except:
        raise SystemExit
    if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.Sage.2nd')):
        raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.Sage.2nd')) if f.endswith(".mzML")])
        raw_dir = '00.rawdata.Sage.2nd'
        Sage(speciesPath_dir, raw_dir, raw_count)
        # submit Slurm and acquire Job ID
        result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.Sage.rerun.sh")], capture_output=True, text=True)
        output = result.stdout.strip()
        job_id = output.split()[-1]  
        print(f"作业已提交，Job ID: {job_id}")
        while True:
            check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id not in check_result.stdout:
                print(f"作业 {job_id} 已完成或退出队列。")
                break
            else:
                print(f"作业 {job_id} 正在运行或等待中...")
            time.sleep(10) 
        
        # Third run
        command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.Sage.3rd"
        try:
            os.system(command)
        except:
            raise SystemExit
        command = "cd " + speciesPath_dir + "/00.rawdata\n" + "ls | grep .mzML > mzML.list\n" + "for mzML in `cat mzML.list`; do\nfilename=$(echo $mzML | sed 's/.mzML$//')\nif [ ! -f \"../05.All_results/01.Sage/${filename}_results.sage.tsv\" ]; then\ncp $mzML ../00.rawdata.Sage.3rd; fi\ndone"
        try:
            os.system(command)
        except:
            raise SystemExit
        if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.Sage.3rd')):
            raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.Sage.3rd')) if f.endswith(".mzML")])
            raw_dir = '00.rawdata.Sage.3rd'
            Sage(speciesPath_dir, raw_dir, raw_count)
            # submit Slurm and acquire Job ID
            result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.Sage.rerun.sh")], capture_output=True, text=True)
            output = result.stdout.strip()
            job_id = output.split()[-1]  
            print(f"作业已提交，Job ID: {job_id}")
            while True:
                check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
                if job_id not in check_result.stdout:
                    print(f"作业 {job_id} 已完成或退出队列。")
                    break
                else:
                    print(f"作业 {job_id} 正在运行或等待中...")
                time.sleep(10) 

def check_job_status(job_id):
    try:
        result = subprocess.run(['squeue', '-h', '-j', str(job_id), '-o', '%T'], capture_output=True, text=True, check=True)
        status = result.stdout.strip()
        if status == 'RUNNING':
            return True, status
        elif status == "PENDING":
            return False, status
        else:
            return None, status
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e}")
        return None, None
    
def cancel_job(job_id):
    try:
        result = subprocess.run(['scancel', str(job_id)], capture_output=True, text=True, check=True)
        print(f"作业 {job_id} 已被终止。")
    except subprocess.CalledProcessError as e:
        print(f"Error running scancel: {e}")
        
def monitor_and_cancel_jobs(job_ids, max_runtime_seconds):
    running_jobs = {}
    while True:
        completes = []
        for job_id in job_ids:
            if job_id in running_jobs:
                is_running, status = check_job_status(job_id)
                completes.append(is_running)
                current_time = time.time()
                elapsed_time = current_time - running_jobs[job_id]['start_time']
                if elapsed_time > max_runtime_seconds:
                    print(f"作业 {job_id} 运行时间超过 {max_runtime_seconds} 秒，正在终止作业...")
                    cancel_job(job_id)
                    del running_jobs[job_id]
                else:
                    print(f"作业 {job_id} 已开始运行，已运行 {elapsed_time:.2f} 秒。")
            else:
                is_running, status = check_job_status(job_id)
                completes.append(is_running)
                if is_running:
                    print(f"作业 {job_id} 已开始运行。")
                    running_jobs[job_id] = {'start_time': time.time()}
                else:
                    print(f"作业 {job_id} 未开始运行，状态为 {status}。")
        print(completes)
        if all(x is None for x in completes):
            print(f"-------------msConver end----------------")
            break
        time.sleep(10)

def main(line, speciesPath):
    print(line.strip()) 
    speciesPath_dir = os.path.join(speciesPath, line)
    print(f"-------------------------正在分析 {line} 物种--------------------------")
    
    # 1.cd workpath
    try:
        os.chdir(speciesPath_dir)
        print(f"当前路径已更改为: {os.getcwd()}")
    except FileNotFoundError:
        print(f"路径不存在: {speciesPath_dir}")
    except PermissionError:
        print(f"没有权限访问该路径: {speciesPath_dir}")
        
    # 2.Count the number of raw files in all project, output the raw file path
    folder_path = os.path.join(os.getcwd(), "pride_raw")
    if os.path.exists(folder_path):
        raw_file_count = 0 
        for root, dirs, files in os.walk(folder_path):
            raw_files = [f for f in files if f.lower().endswith(".raw")]
            raw_file_count += len(raw_files)
        print(f"在 '{folder_path}' 的所有子文件夹中，找到 {raw_file_count} 个 .raw 文件。")
        if raw_file_count != 0:
            command = "ls " + folder_path + "/*/*.raw > pride_raw.txt"
            os.system(command)
        else:
            return
    else:
        print(f"文件夹 '{folder_path}' 不存在。")
        return
        
    # 3.Generate workdir
    generateDir(speciesPath_dir)
    
    # 4.Generate decoy fasta
    command = "cp " + speciesPath_dir + "/*.fasta " + speciesPath_dir + "/02.fasta"
    try:
        os.system(command)
    except:
        raise SystemExit
    generate_decoy_fasta(speciesPath_dir)
    # submit Slurm and acquire Job ID
    result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "00.generate.FP.decoy.fasta.sh")], capture_output=True, text=True)
    output = result.stdout.strip()
    job_id = output.split()[-1]  
    print(f"作业已提交，Job ID: {job_id}")
    while True:
        check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in check_result.stdout:
            print(f"作业 {job_id} 已完成或退出队列。")
            break
        else:
            print(f"作业 {job_id} 正在运行或等待中...")
        time.sleep(10)  
        
    # 5.Move the raw file to the 00.rawdata folder
    command = "for raw in `cat " + speciesPath_dir + "/pride_raw.txt`; do mv $raw "+ speciesPath_dir +"/00.rawdata; done"
    try:
        os.system(command)
    except:
        raise SystemExit
    
    # 6.msConvert run
    # First run
    raw_dir = "00.rawdata"
    msConvert(speciesPath_dir, raw_dir, raw_file_count)
    # submit Slurm and acquire Job ID
    result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.msConvert_rerun.sh")], capture_output=True, text=True)
    output = result.stdout.strip()
    job_id = output.split()[-1]  
    print(f"作业已提交，Job ID: {job_id}")
    job_ids = []
    for job_num in range(1, raw_file_count+1):
        job_array_id = f"{job_id}_{job_num}"  
        job_ids.append(job_array_id)
    monitor_and_cancel_jobs(job_ids, 1200)
    
    # Second msconvert
    command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.mzML.2nd"
    try:
        os.system(command)
    except:
        raise SystemExit
    command = "cd " + speciesPath_dir + "/00.rawdata\n" + "ls | grep .raw > raw.list\n" + "for raw in `cat raw.list`; do\nfilename=$(echo $raw | sed 's/.raw$//')\nif [ ! -f \"${filename}.mzML\" ]; then\ncp $raw ../00.rawdata.mzML.2nd; fi\ndone"
    try:
        os.system(command)
    except:
        raise SystemExit
    if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.mzML.2nd')):
        raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.mzML.2nd')) if f.endswith(".raw")])
        raw_dir = '00.rawdata.mzML.2nd'
        msConvert(speciesPath_dir, raw_dir, raw_count)
        # submit Slurm and acquire Job ID
        job_id = submit_job(os.path.join(speciesPath_dir, "01.msConvert_rerun.sh"))
        print(f"作业已提交，Job ID: {job_id}")
        job_ids = []
        for job_num in range(1, raw_file_count+1):
            job_array_id = f"{job_id}_{job_num}"
            job_ids.append(job_array_id)
        monitor_and_cancel_jobs(job_ids, 1200)

        # Third msconvert
        command = "mkdir -p "+ speciesPath_dir + "/00.rawdata.mzML.3rd"
        try:
            os.system(command)
        except:
            raise SystemExit
        command = "cd " + speciesPath_dir + "/00.rawdata.mzML.2nd\n" + "ls | grep .raw > raw.list\n" + "for raw in `cat raw.list`; do\nfilename=$(echo $raw | sed 's/.raw$//')\nif [ ! -f \"${filename}.mzML\" ]; then\ncp $raw ../00.rawdata.mzML.3rd; fi\ndone"
        try:
            os.system(command)
        except:
            raise SystemExit
        if os.listdir(os.path.join(speciesPath_dir, '00.rawdata.mzML.3rd')):
            raw_count = len([f for f in os.listdir(os.path.join(speciesPath_dir, '00.rawdata.mzML.3rd')) if f.endswith(".raw")])
            raw_dir = '00.rawdata.mzML.3rd'
            msConvert(speciesPath_dir, raw_dir, raw_count)
            # submit Slurm and acquire Job ID
            result = subprocess.run(["sbatch", os.path.join(speciesPath_dir, "01.msConvert_rerun.sh")], capture_output=True, text=True)
            output = result.stdout.strip()
            job_id = output.split()[-1]  
            print(f"作业已提交，Job ID: {job_id}")
            job_ids = []
            for job_num in range(1, raw_file_count+1):
                job_array_id = f"{job_id}_{job_num}" 
                job_ids.append(job_array_id)
            monitor_and_cancel_jobs(job_ids, 1200)

        # move .mzML to 00.rawdata
        command = "mv " + speciesPath_dir + "/00.rawdata.mzML.2nd/*.mzML " + speciesPath_dir + "/00.rawdata\nmv " + speciesPath_dir + "/00.rawdata.mzML.3rd/*.mzML " + speciesPath_dir + "00.rawdata"
        try:
            os.system(command)
        except:
            raise SystemExit
    thread1 = threading.Thread(target = run_FragPipe, args = (speciesPath_dir, ))
    thread2 = threading.Thread(target = run_Sage, args = (speciesPath_dir, ))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

if __name__=="__main__":
    speciesPath = parse.speciesPath
    input = parse.input
    process_with_fixed_speciesPath = partial(main, speciesPath = speciesPath)
    
    with open(os.path.join(speciesPath, input), "r") as file:
        species= [line.strip() for line in file]
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(process_with_fixed_speciesPath, species)
        

