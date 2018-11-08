2018.11.10
http://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/  
위 사이트 참조 필수

# -Review-Trust-Region-Policy-Optimization
> Code is from OpenAI 
[OpenAI baseline Code Review Page] 
> OpenAI 가 제공하는 baseline TRPO 코드에 대해서 import 되는 함수들에 대해서도  
> 추후에 사용할 수 있도록 상세히 서술 정리해보는 페이지.    
> MPI 사용에 대해서도 간단히 정리할 예정.   
> TRPO에 대해서는 https://www.slideshare.net/WoongwonLee/trpo-87165690 이웅원님의 PPT 참조.  



# [MPI 사용법]
> Reference Site! Thank You
> https://pythonprogramming.net/mpi-gather-command-mpi4py-python/   
> 매우매우 좋은 tutorial 사이트입니다.    

## [Scatter & Gather]
![image](https://user-images.githubusercontent.com/40893452/47289310-0b89ef80-d635-11e8-9a32-e5845c828a33.png)  
(1) "comm = MPI.COMM_WORLD" 를 통해서 MPI의 message passing을 사용할 수 있다.  
(2) "size = comm.Get_size()" 를 통해서 생성된 parallel processing을 하기 위한 process가 몇개 생성되었는지를 알 수 있다.  
(3) "rank = comm.Get_rank()" 를 통해서 현재 process 가 몇번째인지를 파악. 보통 rank = 0을 main으로 생각한다.  
(4) "data = comm.scatter(data, root=0)" 를 통해서 "data"라는 이름의 array를 다른 process들에게 전달.  
> scatter는 array를 보낼때 process들에게 하나씩 나누어서 전달합니다.   
> data = [1, 2, 3, 4, 5]라면 각 프로세스는 1, 2, 3, 4, 5를 갖게됩니다.  
(5) "data += 1" 를 통해서 모든 process가 comm.scatter()를 만나고 기다리다 모두 도달하면 다음부터 실행되며, 이때 data는 scatter된 값을 가진다.  
> 2, 3, 4, 5, 6 의 data들을 가지고 있게되고 이를 다시 모아야한다.  
(6) newData = comm.gather(data,root=0) 에서 rank = 0 인 process가 이 값들을 다시 모아서 처리.  
> newData = [2,3,4,5,6]으로 다시 data의 구조대로 array가 만들어진다.  


## [Send & Recv]
![image](https://user-images.githubusercontent.com/40893452/47289723-12b1fd00-d637-11e8-9144-899a8d84878f.png)   
(1) "send(data, dest = 1)" :: 보내고자 하는 data와 destination rank number를 나타낸다.   
(2) "recv(source = 1)" :: send가 있을 때 원하는 rank number로부터의 send를 지정해서 받을 수 있다.   
![image](https://user-images.githubusercontent.com/40893452/47290170-2fe7cb00-d639-11e8-958c-1e4150f9784b.png)   
(1) 이런식으로 dictionary도 주고 받을 수 있다.  
(2) Output in rank 1 :: {'d2' : 42, 'd1' : 55} 와 55 를 출력한다.  
> send(data, dest, tag)를 사용해서 recv(source, tag)시에 tag 순서를 바꾸는 것으로 수신하는 순서도 정할 수 있다.  

## [Bcast]
![image](https://user-images.githubusercontent.com/40893452/47290293-b56b7b00-d639-11e8-9602-4249e3664771.png)  
(1) bcast(data, source) :: source로 부터의 data를 모든 process의 data 값에 넣어준다. recv 같은것 필요 없다.  

# [Code Review]
(1) run_mujoco.py : mujoco gym game 환경에서 GAIL 모델을 학습 & 평가 하는 파일   
(2) adversary.py : Discriminator 의 neural network 구성하는 파일  
(3) model.py : 
(4) mlp_policy.py : 


