import json
import argparse
from tw_rouge import get_rouge



def main(args):
    refs = {}
    
   
    preds = {
        "0_shot":{},
        "1_shot":{},
        "2_shot":{},
        "3_shot":{},
        "4_shot":{}
    }
    

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['output'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            for shot in line['prediction']:
                print(shot)
                preds[shot][line['id']] = line['prediction'][shot].strip() + '\n'
                
    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    print("keys=",keys)
    print("refs=",refs)
    print("preds=",preds)
    rouges = {}
    for shot in preds:
        pred = [preds[shot][key] for key in keys]   
        print(shot,"----------------")   
        print("preds=",pred)
        rouge = get_rouge(pred, refs)
        print(rouge)
        rouges[shot] = rouge
    print(rouges)
    
    with open(args.output_rouge_file, 'w') as json_file:
        json.dump(rouges, json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    parser.add_argument('-o', '--output_rouge_file')
    args = parser.parse_args()
    main(args)
