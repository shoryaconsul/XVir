#include <algorithm>
#include <random>
#include <iomanip> // std::setprecision
#include <vector> // for use of vector
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream> 
#include <numeric>
#include <time.h>
#include <unistd.h>

using namespace std;

/* 
This function parses the FASTQ file line by line and
stores them as a sequence of ints in a vector.
*/
int parseFASTQ(string fastq, vector<vector<int>>& seq_vec){

    string line;
    int k, cnt = 0;
    int newfile = 0;
    bool seq_isnext = true;
    vector<int> seq;

    ifstream infile(fastq.c_str(), ios::in);
    while( getline(infile, line, '\n')){
        if(line[0] == '@'){
            seq_isnext = true;
        }
        else if(seq_isnext){  // Sequence line
            seq.clear();
            for(int i=0; i<line.size(); i++){
                switch(line[i]){
                    case 'A': case 'a':
                        k = 1;
                        break;
                    case 'C': case 'c':
                        k = 2;
                        break;
                    case 'G': case 'g':
                        k = 3;
                        break;
                    case 'T': case 't':
                        k = 4;
                        break;
                }
                seq.push_back(k);
            }
            seq_vec.push_back(seq);
            seq_isnext = false;
            cnt++;
/*             if(cnt < 2){
                for(int i : seq){
                    cout << i ;
                }
                cout << endl;
            cout<<"Parsed line "<< cnt<<endl;
            } */
        }
        else{
            seq_isnext = false;
        }
    }

    cout << "Finished parsing FASTQ file containing " << cnt << " reads." << endl; 

    if(seq_vec.size() == 0) return 1;
    else return 0;
}


/*
This function writes the parsed FASTQ values (as integers) as a space-delimited matrix into
the specified file. 
*/
void write2file(string outname, vector<vector<int>>& seq_vec){
    ofstream outfile;
    outfile.open(outname);
    for(int j=0; j<seq_vec.size(); j++){
        for(int i=0; i<seq_vec[j].size(); i++){
            outfile << seq_vec[j][i] << " ";
        }
        outfile << endl;
    }
    outfile.close();
}

int main(int argc, char* argv[]){

    string fastq_file, out_file;

    const char* const opts_str = ":f:o:";
    int option;
	while((option = getopt(argc, argv, opts_str)) != -1){
        switch(option){
            case 'f':
                fastq_file = optarg;
                break;
            case 'o':
                out_file = optarg;
                break;
            case ':':
                cerr << "Missing argument: " << char(optopt) << endl;
                break;
            case '?':
                cerr << "Invalid option provided: "<< char(optopt) <<endl;
                break;
        }
    }

    // string fastq_file = "human_reads.fq";
    // string out_file = "human_reads_parsed";
    vector<vector<int>> seq_vec;

    parseFASTQ(fastq_file, seq_vec);
    write2file(out_file, seq_vec);
}