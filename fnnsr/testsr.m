
inputdata=zeros(1000,13);
inputlabel=zeros(1000,1);
k=0;
for i=0:9
    n=100*i;
    inputlabel(n+1:n+100,1)=i;
 
    for j=0:99
       filepath=sprintf('E:\\matlab stuff\\speech processing\\Hindi digits\\%d\\%02d.wav',i,j);
       x=(mirmfcc(filepath));
       mirgetdata(x);
       k=k+1;
       inputdata(k,:)=ans';
       end
   
end

 