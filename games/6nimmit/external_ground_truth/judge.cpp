#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <fcntl.h>
#include <cstring>
#include <signal.h>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
using namespace std;

#define MAX_CARDS 104
#define MAX_PLAYERS 10
#define NUM_ROUNDS 10
#define NUM_STACKS 4
#define STACK 5
#define MINIMUM_SCORE -66

#define LOG_FILE "log.txt"
#define PLAYER_OUTPUT "output.txt"

#define TIME_LIMIT 3

int num_players;
int num_cards=104;
int short_game=1;
int num_games=1;

pid_t pid;


string player_file[MAX_PLAYERS];

void overtime(int param) {
	kill(pid,SIGKILL);
	cerr << "Killing process " << pid << endl;
}






class game
{
	ostringstream record; //all information of the current game

	ostringstream duration; //the public information so far



	struct shuffle_str
	{
		int r;
		int card;
		shuffle_str(int c) {
			card=c;
			r=rand();
		}
		bool operator < (const shuffle_str &a) const
		{
			return (r < a.r);
		}
	};

	struct play_str
	{
		int index;
		int card;
		play_str(int i,int c) {
			index=i;
			card=c;
		}
		bool operator < (const play_str &a) const
		{
			return (card < a.card);
		}
	};


	int heads[MAX_CARDS+1];

	int current_player;
	int playing[MAX_PLAYERS];

	void compute_heads() {
		for (int i=1;i<=num_cards;i++) {
			heads[i]=0;
			if (i%5==0) heads[i]+=2;
			if (i%10==0) heads[i]++;
			if (i%11==0) heads[i]+=5;
			if (heads[i]==0) heads[i]=1;

		}

	}

	public:
	int num_players;
	int num_cards;

	int player_cards[MAX_PLAYERS][NUM_ROUNDS];
	int stacks[NUM_STACKS][STACK];
	int c_s[NUM_STACKS];

	int scores[MAX_PLAYERS];

	int current_round;



	game(int p,int c, int s[]) {
		if (c<p*10+4) {
			cerr << "Too few cards" << endl;
			exit(1);
		}

		num_players=p;
		num_cards=c;
		compute_heads();
		for (int i=0;i<num_players;i++)
			scores[i]=s[i];
	}

	game(int p,int c) {
		if (c<p*10+4) {
			cerr << "Too few cards" << endl;
			exit(1);
		}

		num_players=p;
		num_cards=c;
		compute_heads();
		for (int i=0;i<num_players;i++)
			scores[i]=0;
	}

	game() {
		num_players=2;
		num_cards=104;
		compute_heads();
		for (int i=0;i<num_players;i++)
			scores[i]=0;

	}

	int run_program(int p, int a=0) {
		//p stands for index of the current player
		//a=0 stands for selecting a card in a normal step, a=1 stands for choosing a replaced line.
		
		string cmd = player_file[p]; // secondary program you want to run

		int input[2], output[2];

		unlink(PLAYER_OUTPUT);
		
		signal(SIGPIPE, SIG_IGN);
		if (pipe(input)!=0 || pipe(output)!=0) {
			cerr << "Pipe Error!" << endl;
			return 0;
		}

		pid = fork(); // create child process

		if (pid == -1)
		{
			perror("fork");
			exit(1);
		}
		else if (pid == 0) // child process
		{
			dup2(input[0],0);
			dup2(output[1],1);

			close(input[1]);
			close(output[0]);

			if (a==1) execl(cmd.c_str(), cmd.c_str(), "-small", NULL);
			else execl(cmd.c_str(),cmd.c_str(), NULL); // run the command

			perror("execl"); // execl doesn't return unless there is a problem
			cerr << "child process ends." << endl;
			exit(1);
		}
		else // parent process, pid now contains the child pid
		{
			close(input[0]);  
			close(output[1]);

			FILE *child_out = fdopen(output[0], "r");
			FILE *child_in = fdopen(input[1], "w");

			fprintf(child_in, "%d %d\n", p, current_round);
   
			for (int k=0;k<NUM_ROUNDS-current_round;k++)
			  fprintf(child_in, "%d%c", player_cards[p][k], " \n"[k == NUM_ROUNDS-current_round-1]);
		  
			fprintf(child_in, "%s\n", duration.str().c_str());


			// Added 20:39 Aug 31, 16

			for (int k=0;k<num_players;k++)
			  fprintf(child_in, "%d%c", scores[k], " \n"[k == num_players-1]);

			//////
			

			fflush(child_in);

			signal(SIGALRM, overtime);
			alarm(TIME_LIMIT);

			int status;
			while (-1 == waitpid(pid, &status, 0)); // wait for child to complete

			alarm(0);

			if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
			{
				// handle error
				cerr << "process " << cmd << " (pid=" << pid << ") failed" << endl;
			}

			int child_out_int=0;
			if (fscanf(child_out, "%d", &child_out_int) != 1)
			{
			  // cerr << "Try to read output.txt" << endl;
			        ifstream file;
				file.open(PLAYER_OUTPUT);

				if (!file.is_open()) {
					cerr << "No output.txt" << endl;
					// return 0;
				}
				else if (!(file >> child_out_int)) {
					cerr << "NO OUTPUT" << endl;
					// return 0;
				}
				file.close();
				unlink(PLAYER_OUTPUT);
			}

			fclose(child_in);
			fclose(child_out);

			// cout << "The pid is: " << pid << endl;

			return child_out_int;
		}
	}







	void record_log() {
		int i,j;
		record << "CURRENT TABLE:" << endl;
		for (i=0;i<NUM_STACKS;i++) {
			for (j=0;j<c_s[i];j++)
				record << stacks[i][j] << "\t";

			record << endl;
		}

		for (i=0;i<num_players;i++) {
			record << "PLAYER " << i << ": ";
			for (j=0;j<NUM_ROUNDS-current_round;j++)
			{
				record << player_cards[i][j] << "\t";
			}
			record << endl;
			record << "\t SCORE:" << scores[i] << endl;

		}
		record << endl;
	}

	void record_current() {
		for (int i=0;i<NUM_STACKS;i++) {
			for (int j=0;j<STACK;j++)
				duration << stacks[i][j] << " ";

			duration << endl;
		}
	}

	void game_start() {

		int i,j;
		vector<struct shuffle_str> list;
		for (i=1;i<=num_cards;i++) {
			struct shuffle_str *s=new shuffle_str(i);
			list.push_back(*s);
		}
		sort(list.begin(),list.end());
		vector<struct shuffle_str>::iterator it=list.begin();

		memset(stacks,0,sizeof(stacks));

		duration << num_players << " " << num_cards << endl;
		for (i=0;i<NUM_STACKS;i++) {
			stacks[i][0]=it->card;
			c_s[i]=1;
			it++;
			//  duration << stacks[i][0] << " ";

		}

		record_current();
		for (i=0;i<num_players;i++) {
			for (j=0;j<NUM_ROUNDS;j++)
			{
				player_cards[i][j]=it->card;
				it++;
			}
			sort(player_cards[i],player_cards[i]+NUM_ROUNDS,less<int>());
			//    scores[i]=0;
		}

		current_round=0;
		current_player=0;
		time_t rawtime;
		struct tm * timeinfo;

		time (&rawtime);
		timeinfo = localtime (&rawtime);
		record << "Game between ";
		for (i=0;i<num_players;i++)
			record << "player " << i << " (" << player_file[i] << "), ";
		record << endl;
		record << "At time " << asctime(timeinfo) << endl;
		record_log();



	}

	int update_stack(int c,int p) {
		int i;
		int max=0;
		int argmax=0;
		int s=0;   // score
		for (i=0;i<NUM_STACKS;i++)
			if ((stacks[i][c_s[i]-1]<c) && (stacks[i][c_s[i]-1]>max)) {
				max=stacks[i][c_s[i]-1];
				argmax=i;
			}

		if (max==0) {    // c is smaller than the smallest line
			record << "Player " << p << "'s card " << c << " is SMALLER than all lines" << endl;
			argmax=run_program(p,1);

			if ((argmax<0) || (argmax>=NUM_STACKS))
			{
				cerr << "INVALID Replacing line for player " << "p" << endl;
				record << "INVALID Replacing line for player " << "p" << endl;
				argmax=rand()%NUM_STACKS;
			}


			for (i=0;i<c_s[argmax];i++) {
				s += heads[stacks[argmax][i]];
				stacks[argmax][i]=0;
			}      
			c_s[argmax]=1;
			stacks[argmax][0]=c;
		}
		else if (c_s[argmax]>=STACK) {
			for (i=0;i<STACK;i++) {
				s += heads[stacks[argmax][i]];
				stacks[argmax][i]=0;
			}

			c_s[argmax]=1;
			stacks[argmax][0]=c;
		}
		else {
			stacks[argmax][c_s[argmax]]=c;
			c_s[argmax]++;
		}

		return s;
	}


	void update() {
		int i;
		current_round++;
		current_player=0;


		for (i=0;i<num_players;i++)
			duration << playing[i] << " ";
		duration << endl;


		//Sort cards
		vector<struct play_str> list;
		for (i=0;i<num_players;i++) {
			struct play_str *s=new play_str(i,playing[i]);
			list.push_back(*s);

			record << "PLAYER " << i << " gives " << playing[i] << ", ";
		}
		record << endl;
		sort(list.begin(),list.end());

		for (vector<struct play_str>::iterator it=list.begin();it!=list.end();it++) {
			scores[it->index]-=update_stack(it->card,it->index);
		}

	
		record_current();
		record_log();


	}

	void game_play(int c) {
		int card;
		int *p=find(player_cards[current_player],player_cards[current_player]+NUM_ROUNDS-current_round,c);
		if (p!=player_cards[current_player]+NUM_ROUNDS-current_round)
			card=c;
		else {
			cerr << "Invalid card " << c << " for player " << current_player << endl;
			record << "Invalid card " << c << " for player " << current_player << endl;
			card=player_cards[current_player][rand()%(NUM_ROUNDS-current_round)];
		}

		p=player_cards[current_player];
		remove(p,p+NUM_ROUNDS-current_round,card);

		playing[current_player]=card;

		current_player++;

		if (current_player==num_players) update();

	}

	void print_log() {
		ofstream logfile;   //write the log file
		logfile.open(LOG_FILE,ofstream::app);
		logfile << record.str();
		logfile.close();
	}

};


int* single_game(int s[]) {
	int *score=new int[MAX_PLAYERS];
	//  memset(score,0,sizeof(score));

	int i,p,c;
	game *game1;
	if (s==NULL)
		game1=new game(num_players,num_cards);
	else
		game1=new game(num_players,num_cards,s);
	game1->game_start();

	for (i=0;i<NUM_ROUNDS;i++) {
		for (p=0;p<num_players;p++) {
			c=game1->run_program(p);
			game1->game_play(c);
		}
	}

	game1->print_log();

	for (p=0;p<num_players;p++)
		score[p]=game1->scores[p];

	delete(game1);

	return score;
}



float* compute_points(int s[]) {
	int max_score = *max_element(s,s+num_players);
	int min_score = *min_element(s,s+num_players);
	int num_max=0;
	int num_min=0;
	int i;
	float *f=new float[MAX_PLAYERS];
	for (i=0;i<num_players;i++) {
		if (s[i]==max_score) num_max++;
		if (s[i]==min_score) num_min++;
		f[i]=0.0;
	}
	if ((num_players>2) && (max_score==min_score)) return f;
	
	for (i=0;i<num_players;i++) {
		if (s[i]==max_score) f[i]+=(1.0/num_max);
		if ((num_players>2) && (s[i]==min_score)) f[i]-=(1.0/num_min);
	}
	return f;
}




int main(int argc, char *argv[])
{
	srand(time(0));// set random seed
	rand();

	if (argc<3) return 0;
	int i,g;

	num_players=0;
	for (i=1;i<argc;i++)
		if (argv[i][0]=='-') {

			if (i>=argc-1) {
				cerr << "Parameter Error!" << endl;
				return 0;
			}

			int j=atoi(argv[i+1]);
			switch (argv[i][1]) {
				case 'c': num_cards=j;break;
				case 's': short_game=j;break;
				case 'g': num_games=j;break;
			}
			i++;
		}
		else {
			player_file[num_players]=(string)(argv[i]);
			num_players++;
		}


	cout << "Number of Players: " << num_players << endl;
	cout << "Number of Cards: " << num_cards << endl;
	cout << "Short game? " << short_game << endl;
	cout << "Number of games: " << num_games << endl;
	for (i=0; i<num_players; i++) cout << "Player " << i << ": " << player_file[i] << endl;

	float points[MAX_PLAYERS];
	for (i=0; i<num_players; i++)
		points[i]=0.0;
	for (g=0;g<num_games; g++)
	{
		if (short_game) {
			int* score=single_game(NULL);
			float* p;
			cout << "Game " << g << " scores:" << endl << "\t";
			for (i=0; i<num_players; i++)
				cout << score[i] << " ";
			cout << endl;
			p=compute_points(score);
			cout << "Players' points:" << endl << "\t";
			for (i=0; i<num_players; i++) {

				cout << p[i] << " ";
				points[i]+=p[i];
			}
			cout << endl;
		}
		else {
			int score[MAX_PLAYERS];
			memset(score,0,sizeof(score));
			int* s;
			while (*min_element(score,score+num_players)>MINIMUM_SCORE) {
				s=single_game(score);
				for (i=0; i<num_players; i++)
					score[i]=s[i];

				cout << "Game " << g << " scores:" << endl << "\t";
				for (i=0; i<num_players; i++)
					cout << score[i] << " ";
				cout << endl;
			}

			float *p;
			p=compute_points(score);
			cout << "Players' final points:" << endl << "\t";
			for (i=0; i<num_players; i++) {

				cout << p[i] << " ";
				points[i]+=p[i];
			}
			cout << endl;
		}

	}
	cout << endl;
	cout << "TOTAL POINTS: " << endl;
	for (i=0; i<num_players; i++)
		cout << "PLAYER " << i << ": " << points[i] << endl;



	return 0;
}

