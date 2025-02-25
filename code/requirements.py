# It installs various libraries and packages needed
# by the sci4_reorder_edge_cosine-310.py and the 
# sci4_reorder_edge_plus_sbert-310.py files

import time
import subprocess
import sys
import socket
import os
from typing import Union

def install_package(package_name: str) -> Union[None, str]:
    '''
    Install a package using pip and handle potential installation errors

    Notes:
        Assumes that the user will exercise due caution with regards to
        typing the name of the desired package properly

    Arguments:
        package_name (str): The name of the package to be installed

    Returns:
        None if there are no installation errors
        e (str), if there is an installation error
    '''
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f'\033[1;32m{package_name} installed successfully\033[0m')
        return None
    except Exception as e:
        print(f'\033[1;31mSome error occurred while installing {package_name}:\n{e}\033[0m')
        if package_name == 'textacy':
            print(f'\033[1;31mTrying to install {package_name} using Admin privileges\033[0m')
            try:
                subprocess.check_call(['runas', '/user:hp', f'{sys.executable} -m pip install {package_name}'])
            except Exception as e2:
                print(f'\033[1;31mAnother error occurred while installing {package_name}:\n{e2}\033[0m')
        return f'Some error occurred while installing {package_name}:\n{e}'

def run_safety_checks() -> None:
    """
    Run safety checks to ensure smooth running of the program.

    Notes:
        Checks conducted:
            RAM check (minimum 8 GB)
            Port check (9000)
            Disk space check (minimum 4 GB)
        Exits program if checks give in

    Returns:
        None
    """
    print(f'\033[1;33mRunning Safety Checks before starting...\033[0m')

    import psutil
    # psutil gets installed later

    # Check available RAM in GB
    total_memory = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)
    if total_memory >= 7: print(f'\033[1;32mGreat. The system has sufficient RAM of at least 8 GB. Moving on to the next requirement.\033[0m')
    else: print(f'\033[1;31mThe system must have at least 8 GB RAM. {total_memory} GB RAM is present. Try running this code on Google Colab.\033[0m'); exit()
    
    # Check if port 9000 is available
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('localhost', 9000))
    if result == 0: 
        print('\033[0;31mPort 9000 is in use. Please close the process\033[0m')
        print('\033[0;33mFirst run the following command\033[0m')
        print('\033[0;34mnetstat -ano | findstr :9000\033[0m')
        print('\033[0;33mOn running that command, a similar output may come as follows\033[0m')
        print('\033[0;34mTCP    0.0.0.0:9000           0.0.0.0:0              LISTENING       14672\033[0m')
        print('\033[1;33mUse the number on the extreme right (in this example, 14672) and run a command similar to\033[0m')
        print('\033[0;34mtaskkill /PID 14672 /F\033[0m')
        exit()
    else: print(f'\033[1;32mGreat. Port 9000 is free. Moving on to the next requirement.\033[0m')

    # Check disk space of the current drive
    current_dir = os.getcwd()
    current_drive = os.path.splitdrive(current_dir)[0]
    # If no drive is detected (e.g., in Unix-like systems), use the root directory
    if not current_drive:
        current_drive = '/'
    disk_usage = psutil.disk_usage(current_drive)
    free_space_gb = round(disk_usage.free / (1024.0 ** 3), 2)
    if free_space_gb > 4:
        print(f'\033[1;32mGreat. The working disk {current_drive} has {free_space_gb} GB left, which is more than the minimum 4GB.\033[0m')
        print(f'\033[1;32mWe are good to proceed. The main programs can be run.\033[0m')
    else:
        print(f'\033[1;31mThe working disk {current_drive} has {free_space_gb} GB left, which is less than the minimum of 4GB.\033[0m')
        exit()

def main():
    print(f'\033[1;32m\
    Welcome to the Requirements Program. This program ensures all requirements are fulfilled to enable a smooth run')

    # Confirm from user if Java is installed or not
    confirmation = input('\033[0;33m \
    \nIs Java installed in your system?\
    \nBefore answering, consider the following points: \
    \n1. Navigate to C:/ProgramFiles \
    \n2. If Java folder is present, it means Java is installed \
    \n3. If not, it means Java needs to be installed in which case, go to: https://www.oracle.com/java/technologies/javase-downloads.html\
    \n4. Download whichever version of Java you want, based on your device\'s specifications like OS \
    \n5. During the download, ensure that the download path set is C:/ProgramFiles/Java \
    \n6. Now set the PATH variable. Navigate to C:/ProgramFiles/Java/jdk-<version>/bin and copy the path.\
    \n7. Now go to Settings > System > About > Advanced System Settings\
    \n8. The System Properties Dialog Box opens up. Click on the Environment Variables... button.\
    \n9. The Environment Variables Dialog Box opens up. Under System Variables, double-click on Path\
    \n10. Press the New button and paste the copied path. Click OK and close all dialog boxes.\
    \n11. Even if you hd a pre-existing version of Java, ensure that it is present in C:/ProgramFiles/ folder\
          and that the full path is C:/ProgramFiles/Java/jdk-<version>/bin and that the same path is set as one\
          of the values of Path in Environment Variables.\
    \nIf you cofirm doing all this, enter "Y" else "N"\nEnter: ')
    if confirmation == "Y": print('\033[1;32mWe are good to proceed with the installation\033[0m'); time.sleep(3)
    else: print('\033[1;31mPlease ensure Java is installed properly in accordance with the instructions. \033[0m'); exit()

    # Confirm from user if Virtual Environment has been created and activated or not
    confirmation = input('\033[0;33mDid you first create and activate a virtual environment?\nIf yes, enter "Y" else "N"\nEnter: \033[0m')
    if confirmation == "Y": print('\033[1;32mGreat. Moving on to the next requirement.\033[0m'); time.sleep(3)
    else: print('\033[1;31mPlease create a virtual environment to ensure that no dependency clash occurs\033[0m'); exit()
    
    # Installation status tracking
    installation_error_messages = []

    # 1. Stanza
    print('\033[1;33mInstalling Stanza...\033[0m')
    installation_error_messages.append(install_package('stanza'))
    time.sleep(1)

    import stanza

    # 2. Textacy
    print('\033[1;33mUninstalling Numpy 2.2.0 to avoid version conflict...(You will be asked to proceed. Press Y.)\033[0m')

    result = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'numpy==2.2.0', '-y'],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f'\033[1;32mnumpy==2.2.0 uninstalled successfully\033[0m')
    else:
        if "Successfully uninstalled numpy-2.2.0" in result.stdout:
            print(f'\033[1;32mnumpy==2.2.0 uninstalled successfully\033[0m')
        else:
            print(f'\033[1;31mSome error occurred while uninstalling numpy==2.2.0:\n{result.stderr}\033[0m')
        

    print('\033[1;33mInstalling Jellyfish 1.1.0 (a prerequisite for Textacy which needs to be explicitly specified)...\033[0m')
    installation_error_messages.append(install_package('jellyfish==1.1.0'))
    
    print('\033[1;33mInstalling Textacy...\033[0m')
    installation_error_messages.append(install_package('textacy'))
    time.sleep(1)

    # 3. NLTK
    print('\033[1;33mInstalling NLTK...\033[0m')
    installation_error_messages.append(install_package('nltk'))
    time.sleep(1)

    # 4. CoreNLP
    print('\033[1;33mInstalling CoreNLP...\033[0m')
    try:
        corenlp_dir = './corenlp'
        stanza.install_corenlp(dir=corenlp_dir)
        print('\033[1;32mCoreNLP installed successfully\033[0m')
        installation_error_messages.append(None)
    except Exception as e:
        print(f'\033[1;31mSome error occurred while installing CoreNLP:\n{e}\033[0m')
        installation_error_messages.append(f'Some error occurred while installing CoreNLP:\n{e}')
    time.sleep(1)

    # 5. SpaCy and en_core_web_sm
    print('\033[1;33mInstalling SpaCy...\033[0m')
    installation_error_messages.append(install_package('spacy'))
    print('\033[1;33mInstalling en_core_web_sm...\033[0m')
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        print(f'\033[1;32men_core_web_sm installed successfully\033[0m')
        installation_error_messages.append(None)
    except Exception as e:
        print(f'\033[1;31mSome error occurred while installing en_core_web_sm:\n{e}\033[0m')
        installation_error_messages.append(e)
    time.sleep(1)

    # 6. Sentence Transformers
    print('\033[1;33mInstalling Sentence Transformers...\033[0m')
    installation_error_messages.append(install_package('sentence_transformers'))
    time.sleep(1)

    # 7. Pandas
    print('\033[1;33mInstalling Pandas...\033[0m')
    installation_error_messages.append(install_package('pandas'))
    time.sleep(1)

    # 8. Psutil
    print('\033[1;33mInstalling Psutil...\033[0m')
    installation_error_messages.append(install_package('psutil'))
    time.sleep(1)

    # 9. OpenPyxl
    print('\033[1;33mInstalling OpenPyxl...\033[0m')
    installation_error_messages.append(install_package('openpyxl'))
    time.sleep(1)

    flag = True
    counter = 1
    for installation_error_message in installation_error_messages:
        if installation_error_message is not None:
            print(f'\033[1;35mMessage {counter}:\n{installation_error_message}') 
            flag = False
            counter += 1

    if flag: print('\033[1;32mALL INSTALLATIONS SUCCESSFUL')
    else: 
        print(f'\033[1;31mSome installation did not go as per plan. Check error messages')
        print(f'\033[1;35mSome things you can do:\n\
        1. Check the terminal stack trace and find out which library is causing the issue. For example you will notice\n \
           that jellifish==1.1.0 is being explicitly installed just before textacy in this code. Why? This is because\n \
           when the installation for textacy runs, one of the required packages which automatically gets installed\n \
           is jellyfish. Now the latest version of jellyfish is more than 1.1.0, which is where the problem came\n \
           as jellyfish 1.1.2 requires Rust, and the laptop did not have it nor was there any intention to install\n \
           Rust. So instead of letting textacy install procedure automatically install a wrong version of jellyfish\n \
           (1.1.2), the correct version of jellyfish (1.1.0) is installed manually. Similarly in the future some\n \
           dependency changes can cause similar errors. So it can be resolved like this.\n \
        2. Alternatively, try running the code again. It could work.')

    run_safety_checks()

if __name__ == "__main__":
    main()

'''
FOR DEBUGGING PURPOSES
In case in the future some installation is not working due to a subdependency issue (like it was with jellyfish),
find which subdependency is causing the problem by going through the terminal stack trace. After identifying the 
package which is causing the issue, manually install it by enforcing the correct version (like for jellyfish).

For stanza:
Requirement already satisfied: stanza in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (1.9.2)
Requirement already satisfied: emoji in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) (2.14.0)
Requirement already satisfied: numpy in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) (2.0.2)
Requirement already satisfied: protobuf>=3.15.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from 
stanza) (5.29.1)
Requirement already satisfied: requests in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) 
(2.32.3)
Requirement already satisfied: networkx in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) 
(3.4.2)
Requirement already satisfied: torch>=1.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) (2.5.1)
Requirement already satisfied: tqdm in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from stanza) (4.67.1)
Requirement already satisfied: filelock in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.3.0->stanza) (3.16.1)
Requirement already satisfied: typing-extensions>=4.8.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.3.0->stanza) (4.12.2)
Requirement already satisfied: jinja2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.3.0->stanza) (3.1.4)
Requirement already satisfied: fsspec in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.3.0->stanza) (2024.10.0)
Requirement already satisfied: sympy==1.13.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.3.0->stanza) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sympy==1.13.1->torch>=1.3.0->stanza) (1.3.0)
Requirement already satisfied: charset-normalizer<4,>=2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->stanza) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->stanza) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->stanza) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->stanza) (2024.8.30)
Requirement already satisfied: colorama in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from tqdm->stanza) (0.4.6)
Requirement already satisfied: MarkupSafe>=2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from jinja2->torch>=1.3.0->stanza) (3.0.2)

textacy:
Requirement already satisfied: textacy in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (0.13.0)
Requirement already satisfied: cachetools>=4.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (5.5.0)
Requirement already satisfied: catalogue~=2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (2.0.10)
Requirement already satisfied: cytoolz>=0.10.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (1.0.0)
Requirement already satisfied: floret~=0.10.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (0.10.5)
Requirement already satisfied: jellyfish>=0.8.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from 
textacy) (1.1.0)
Requirement already satisfied: joblib>=0.13.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (1.4.2)
Requirement already satisfied: networkx>=2.7 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (3.4.2)
Requirement already satisfied: numpy>=1.17.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (2.0.2)
Requirement already satisfied: pyphen>=0.10.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (0.17.0)
Requirement already satisfied: requests>=2.10.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from 
textacy) (2.32.3)
Requirement already satisfied: scipy>=1.8.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (1.14.1)
Requirement already satisfied: scikit-learn>=1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (1.6.0)
Requirement already satisfied: spacy~=3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (3.8.3)
Requirement already satisfied: tqdm>=4.19.6 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from textacy) (4.67.1)
Requirement already satisfied: toolz>=0.8.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from cytoolz>=0.10.1->textacy) (1.0.0)
Requirement already satisfied: charset-normalizer<4,>=2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests>=2.10.0->textacy) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests>=2.10.0->textacy) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests>=2.10.0->textacy) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests>=2.10.0->textacy) (2024.8.30)
Requirement already satisfied: threadpoolctl>=3.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from scikit-learn>=1.0->textacy) (3.5.0)
Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (3.0.12)
Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (1.0.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (1.0.11)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (2.0.10)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (3.0.9)
Requirement already satisfied: thinc<8.4.0,>=8.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (8.3.2)
Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (1.1.3)
Requirement already satisfied: srsly<3.0.0,>=2.4.3 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (2.5.0)
Requirement already satisfied: weasel<0.5.0,>=0.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (0.4.1)
Requirement already satisfied: typer<1.0.0,>=0.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (0.15.1)
Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (2.10.3)
Requirement already satisfied: jinja2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (3.1.4)
Requirement already satisfied: setuptools in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (65.5.0)
Requirement already satisfied: packaging>=20.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (24.2)
Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy~=3.0->textacy) (3.5.0)
Requirement already satisfied: colorama in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from tqdm>=4.19.6->textacy) (0.4.6)
Requirement already satisfied: language-data>=1.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from langcodes<4.0.0,>=3.2.0->spacy~=3.0->textacy) (1.3.0)
Requirement already satisfied: annotated-types>=0.6.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages 
(from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy~=3.0->textacy) (0.7.0)
Requirement already satisfied: pydantic-core==2.27.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy~=3.0->textacy) (2.27.1)
Requirement already satisfied: typing-extensions>=4.12.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy~=3.0->textacy) (4.12.2)
Requirement already satisfied: blis<1.1.0,>=1.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from thinc<8.4.0,>=8.3.0->spacy~=3.0->textacy) (1.0.2)
Requirement already satisfied: confection<1.0.0,>=0.0.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from thinc<8.4.0,>=8.3.0->spacy~=3.0->textacy) (0.1.5)
Requirement already satisfied: click>=8.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (8.1.7)
Requirement already satisfied: shellingham>=1.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (1.5.4)
Requirement already satisfied: rich>=10.11.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (13.9.4)
Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from weasel<0.5.0,>=0.1.0->spacy~=3.0->textacy) (0.20.0)
Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from weasel<0.5.0,>=0.1.0->spacy~=3.0->textacy) (7.0.5)
Requirement already satisfied: MarkupSafe>=2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from jinja2->spacy~=3.0->textacy) (3.0.2)
Requirement already satisfied: marisa-trie>=1.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy~=3.0->textacy) (1.2.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (2.18.0)
Requirement already satisfied: wrapt in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy~=3.0->textacy) (1.17.0)
Requirement already satisfied: mdurl~=0.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy~=3.0->textacy) (0.1.2)

nltk:
Requirement already satisfied: nltk in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (3.9.1)
Requirement already satisfied: click in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from nltk) (8.1.7)
Requirement already satisfied: joblib in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from nltk) (2024.11.6)
Requirement already satisfied: tqdm in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from nltk) (4.67.1)
Requirement already satisfied: colorama in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from click->nltk) (0.4.6)

spacy:
Requirement already satisfied: spacy in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (3.8.3)
Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (3.0.12)
Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (1.0.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (1.0.11)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.0.10)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (3.0.9)
Requirement already satisfied: thinc<8.4.0,>=8.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (8.3.2)
Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (1.1.3)
Requirement already satisfied: srsly<3.0.0,>=2.4.3 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.5.0)
Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.0.10)
Requirement already satisfied: weasel<0.5.0,>=0.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (0.4.1)
Requirement already satisfied: typer<1.0.0,>=0.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (0.15.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (4.67.1)
Requirement already satisfied: numpy>=1.19.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.32.3)
Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (2.10.3)
Requirement already satisfied: jinja2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (3.1.4)
Requirement already satisfied: setuptools in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (65.5.0)
Requirement already satisfied: packaging>=20.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (24.2)
Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from spacy) (3.5.0)
Requirement already satisfied: language-data>=1.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from langcodes<4.0.0,>=3.2.0->spacy) (1.3.0)
Requirement already satisfied: annotated-types>=0.6.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages 
(from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.7.0)
Requirement already satisfied: pydantic-core==2.27.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (2.27.1)
Requirement already satisfied: typing-extensions>=4.12.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (4.12.2)
Requirement already satisfied: charset-normalizer<4,>=2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2024.8.30)
Requirement already satisfied: blis<1.1.0,>=1.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from thinc<8.4.0,>=8.3.0->spacy) (1.0.2)
Requirement already satisfied: confection<1.0.0,>=0.0.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from thinc<8.4.0,>=8.3.0->spacy) (0.1.5)
Requirement already satisfied: colorama in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from tqdm<5.0.0,>=4.38.0->spacy) (0.4.6)
Requirement already satisfied: click>=8.0.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy) (8.1.7)
Requirement already satisfied: shellingham>=1.3.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy) (1.5.4)
Requirement already satisfied: rich>=10.11.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from typer<1.0.0,>=0.3.0->spacy) (13.9.4)
Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from weasel<0.5.0,>=0.1.0->spacy) (0.20.0)
Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from weasel<0.5.0,>=0.1.0->spacy) (7.0.5)
Requirement already satisfied: MarkupSafe>=2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from jinja2->spacy) (3.0.2)
Requirement already satisfied: marisa-trie>=1.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy) (1.2.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (2.18.0)
Requirement already satisfied: wrapt in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy) (1.17.0)
Requirement already satisfied: mdurl~=0.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (0.1.2)

en_core_web_sm
en_core_web_sm==3.8.0

Sentence Transformers
Requirement already satisfied: sentence_transformers in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (3.3.1)
Requirement already satisfied: transformers<5.0.0,>=4.41.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (4.47.0)
Requirement already satisfied: tqdm in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (4.67.1)
Requirement already satisfied: torch>=1.11.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (2.5.1)
Requirement already satisfied: scikit-learn in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (1.6.0)
Requirement already satisfied: scipy in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (1.14.1)
Requirement already satisfied: huggingface-hub>=0.20.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (0.26.5)
Requirement already satisfied: Pillow in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sentence_transformers) (11.0.0)
Requirement already satisfied: filelock in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from huggingface-hub>=0.20.0->sentence_transformers) (3.16.1)
Requirement already satisfied: fsspec>=2023.5.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from 
huggingface-hub>=0.20.0->sentence_transformers) (2024.10.0)
Requirement already satisfied: packaging>=20.9 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from huggingface-hub>=0.20.0->sentence_transformers) (24.2)
Requirement already satisfied: pyyaml>=5.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from huggingface-hub>=0.20.0->sentence_transformers) (6.0.2)
Requirement already satisfied: requests in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from huggingface-hub>=0.20.0->sentence_transformers) (2.32.3)
Requirement already satisfied: typing-extensions>=3.7.4.3 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from huggingface-hub>=0.20.0->sentence_transformers) (4.12.2)
Requirement already satisfied: networkx in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.11.0->sentence_transformers) (3.4.2)
Requirement already satisfied: jinja2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.11.0->sentence_transformers) (3.1.4)
Requirement already satisfied: sympy==1.13.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from torch>=1.11.0->sentence_transformers) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from sympy==1.13.1->torch>=1.11.0->sentence_transformers) (1.3.0)
Requirement already satisfied: colorama in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from tqdm->sentence_transformers) (0.4.6)
Requirement already satisfied: numpy>=1.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from transformers<5.0.0,>=4.41.0->sentence_transformers) (2.0.2)
Requirement already satisfied: regex!=2019.12.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from transformers<5.0.0,>=4.41.0->sentence_transformers) (2024.11.6)
Requirement already satisfied: tokenizers<0.22,>=0.21 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages 
(from transformers<5.0.0,>=4.41.0->sentence_transformers) (0.21.0)
Requirement already satisfied: safetensors>=0.4.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from transformers<5.0.0,>=4.41.0->sentence_transformers) (0.4.5)
Requirement already satisfied: joblib>=1.2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from scikit-learn->sentence_transformers) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from scikit-learn->sentence_transformers) (3.5.0)
Requirement already satisfied: MarkupSafe>=2.0 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from jinja2->torch>=1.11.0->sentence_transformers) (3.0.2)
Requirement already satisfied: charset-normalizer<4,>=2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->huggingface-hub>=0.20.0->sentence_transformers) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->huggingface-hub>=0.20.0->sentence_transformers) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->huggingface-hub>=0.20.0->sentence_transformers) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from requests->huggingface-hub>=0.20.0->sentence_transformers) (2024.8.30)

Pandas (should not be required, but still)

Requirement already satisfied: pandas in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (2.2.3)
Requirement already satisfied: numpy>=1.23.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pandas) (2.0.2)
Requirement already satisfied: python-dateutil>=2.8.2 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages 
(from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pandas) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from pandas) (2024.2)
Requirement already satisfied: six>=1.5 in d:\academics folder\internships\27aug-__24 copyright and research\virtualenv\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)

'''