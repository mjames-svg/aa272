from pathlib import Path
from phase5_event_integration import run
from phase5_paired_audit import run as paired

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    run(root)
    paired(root)
