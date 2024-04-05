from model import XVir
from argparse import ArgumentParser
from utils.dataset import inferenceDataset
import torch
import numpy as np
from tqdm import tqdm

def parse_fasta(filename):
    base2int = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    reads = []
    labels = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            elif line[0] == '>':  # Parse sequence data
                read_header = line[1:].rstrip()
                read_seq = f.readline().rstrip().upper()
                if 'N' in read_seq:
                    continue
                read_seq = np.array([base2int[base] for base in read_seq])
                labels.append(read_header)
                reads.append(read_seq)
            else:
                pass # For completeness
    return np.array(reads), np.array(labels)

def main(args):
    model = XVir(args.read_len, args.ngram, args.model_dim, args.num_layers, 0.1)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("Model loaded")
    if args.cuda:
        model = model.to('cuda')
    reads, labels = parse_fasta(args.input)
    dataset = inferenceDataset(args, reads, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Fasta loaded, Predicting...")
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            if args.cuda:
                x = x.to('cuda')
            labels.extend(y)
            outputs = model(x)
            pred = torch.sigmoid(outputs.detach().to('cpu'))
            preds.extend(pred)
    with open(args.input + '.output.txt', 'w') as f:
        for i in range(len(preds)):
            f.write(str(labels[i]) + '\t' + str(preds[i].item()) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--read_len', type=int, default=150)
    parser.add_argument('--ngram', type=int, default=6)
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
    