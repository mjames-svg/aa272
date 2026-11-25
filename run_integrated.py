"""Offline final-package rebuild from bundled inputs and results."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def go(file,*args):
    print('Running:',file,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/file),*args],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rebuild-stadium',action='store_true')
    parser.add_argument('--verify-originals',action='store_true')
    args=parser.parse_args()
    go('phase10_geometry_stadium.py',*(() if args.rebuild_stadium else ('--plots-only',)))
    go('test_phase10.py')
    if args.verify_originals:
        for file in ['self_check.py','test_phase2.py','test_phase3.py','test_phase4.py','test_phase5.py',
          'test_phase7.py','test_phase8.py','test_phase9.py','test_final_analysis.py','verify_final_package.py']:
            go(file)
    go('build_final_story.py')
    print('Integrated research report, presentation, figure atlas and data validation complete.')

if __name__=='__main__':main()
