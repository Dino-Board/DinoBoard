#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <algorithm>
using namespace std;


#define MAX_CARDS 104
#define MAX_PLAYERS 10
#define NUM_ROUNDS 10
#define NUM_STACKS 4
#define STACK 5

#define LOG_FILE "log.txt"


#define TIME_LIMIT 2


int player_index;
int current_round;
int num_players;
int num_cards;

int stacks[NUM_STACKS][STACK];
int c_s[NUM_STACKS];
int stack_heads[NUM_STACKS];


int cards[NUM_ROUNDS];
int num_c;

int heads[MAX_CARDS+1];

int small=0;


void compute_heads() {
  heads[0]=0;
  for (int i=1;i<=num_cards;i++) {
    heads[i]=0;
    if (i%5==0) heads[i]+=2;
    if (i%10==0) heads[i]++;
    if (i%11==0) heads[i]+=5;
    if (heads[i]==0) heads[i]=1;
    
  }
}

double evaluate(int index) {
  int i;
  int card=cards[index];
  int max=0;
  int argmax=0;
  double e; 
  for (i=0;i<NUM_STACKS;i++)
      if ((stacks[i][c_s[i]-1]<card) && (stacks[i][c_s[i]-1]>max)) {
	max=stacks[i][c_s[i]-1];
	argmax=i;
      }

  if (max==0) e= (1.0- card/ num_cards) * (*min_element(stack_heads,stack_heads+NUM_STACKS));
  else if (c_s[argmax]==STACK) e= (1.0- (card-stacks[argmax][STACK-1]) / num_cards) * stack_heads[argmax];
  else {
    e= ((-0.1+card-stacks[argmax][c_s[argmax]-1]) /num_cards )/ (STACK-c_s[argmax]);
    int next_card;
    if (index>=num_c-1) next_card=num_cards+1;
    else next_card=cards[index+1];
    if (c_s[argmax]==STACK-1) e+=(1.0-(next_card-card))/num_cards;
    else e-=(1.0-(next_card-card))/num_cards;
  }
  
  
  return e;
}
  

int main(int argc, char *argv[]) {
  srand(clock());
  int i,j,k,k1,c;
    
  small=0;
  if  ((argc==2) &&  (argv[1][1]=='s')) small=1;

  cin >> player_index;
  cin >> current_round;
  num_c=NUM_ROUNDS-current_round;

    
  for (i=0;i<num_c;i++)
    cin >> cards[i];
 
  cin >> num_players;
  cin >> num_cards;
  

  compute_heads(); 
  if (small) current_round--;
  
  for (i=0;i<=current_round;i++)
    {
      memset(stacks,0,sizeof(stacks));
      memset(stack_heads,0,sizeof(stack_heads));
      for (j=0;j<NUM_STACKS;j++) {
	k1=0;
	for (k=0;k<STACK;k++)
	  {
	    cin >> c;
	    if (c>0) {
	      stacks[j][k]=c;
	      stack_heads[j]+=heads[c];
	      k1++;
	      }
	  }
	c_s[j]=k1;
      }
      if ((i<current_round) || (small==1))
	  for (j=0;j<num_players;j++)
	    cin >> c;
      
    }
   
  
  
  
  
  if (small) {
    
      int min_heads=*min_element(stack_heads,stack_heads+NUM_STACKS);
      for (i=0;i<NUM_STACKS;i++)
	if (stack_heads[i]==min_heads) {
	  cout << i << endl;
	  break;
	}
  } else {
    
    double min=1000.0;
    int s_card=0;
    double e;
      for (i=0;i<num_c;i++) {
	e=evaluate(i);
	
	if (e<min) {
	  min=e;
	  
	  s_card=cards[i];
	}
      }
      cout << s_card << endl;
  }
  
  return 0;
  
}

