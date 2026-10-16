"""Archive a committed v6 documentation tag and verify its shareable contents."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v6.1.0');args=parser.parse_args()
    tag=args.tag
    commit=git('rev-parse',tag+'^{}').decode().strip()
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    archive=out/f'kaggle-houseprice-{tag}.zip';prefix=f'kaggle-houseprice-{tag}/'
    subprocess.run(['git','archive','--format=zip','--prefix='+prefix,'--output='+str(archive),tag],cwd=ROOT,check=True)
    expected=set(git('ls-tree','-r','--name-only',tag).decode().splitlines())
    forbidden={'.git','.venv','.venv-tabpfn','.runtime','.cache','delivery'}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names={n[len(prefix):] for n in z.namelist() if not n.endswith('/')}
        assert names==expected
        assert not any(n.split('/')[0] in forbidden for n in names)
        required=['README.md','docs/SHARING_GUIDE.md','docs/house_prices_v6_final_report.html',
                  'output/pdf/house_prices_v6_final_report.pdf','submission_v5.csv','submission_v6.csv',
                  'scripts/predict_v6.py','scripts/run_tabpfn.sh','src/house_prices_v6.py','src/house_prices.py',
                  'scripts/tabpfn_variants_v6.py','requirements-tabpfn.lock.txt','requirements-tabpfn-portable.txt',
                  'data/raw/train.csv','data/raw/test.csv','data/raw/data_description.txt',
                  'experiments/v6/fit_context.joblib','experiments/v6/selection.json',
                  'experiments/v6/online_result.json','experiments/v6/submission_audit.json',
                  'experiments/v6/pretrained/tabpfn-v2-regressor.ckpt','experiments/v6/pretrained/LICENSE.txt']
        assert set(required).issubset(names)
        checks={n:sha(z.read(prefix+n)) for n in required}
        online=json.loads(z.read(prefix+'experiments/v6/online_result.json'))
        selection=json.loads(z.read(prefix+'experiments/v6/selection.json'))
        assert online['public_log_rmse']==.11212
        assert checks['submission_v6.csv']==online['submission_sha256']==selection['submission_sha256']
        assert checks['experiments/v6/pretrained/tabpfn-v2-regressor.ckpt']==selection['pretrained_sha256']
        assert checks['experiments/v6/fit_context.joblib']==selection['fit_context_sha256']
        assert checks['src/house_prices_v6.py']==selection['pipeline_source_sha256']
        assert checks['src/house_prices.py']==selection['feature_source_sha256']
        assert checks['scripts/tabpfn_variants_v6.py']==selection['quality_transform_source_sha256']
        assert checks['requirements-tabpfn.lock.txt']==selection['requirements_sha256']
        uncompressed=sum(i.file_size for i in z.infolist())
    deliverables={p.name:{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
                  for p in [archive,ROOT/'output/pdf/house_prices_v6_final_report.pdf',ROOT/'docs/house_prices_v6_final_report.html']}
    manifest={'tag':tag,'commit':commit,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'public_log_rmse_user_reported':.11212,'final_adopted_version':'v6',
              'archive':str(archive.relative_to(ROOT)),'archive_files':len(expected),'uncompressed_bytes':uncompressed,
              'zip_crc_check_passed':True,'all_files_match_git_tree':True,'runtime_and_git_directories_excluded':True,
              'critical_file_sha256':checks,'deliverables':deliverables,
              'release_url':'https://github.com/WANGDAKEKE/kaggle-houseprice/releases/tag/'+tag}
    (out/'share_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'tag':tag,'commit':commit,'files':len(expected),'zip_bytes':archive.stat().st_size,
                     'zip_sha256':deliverables[archive.name]['sha256'],'checks_passed':True},indent=2))


if __name__=='__main__':main()
