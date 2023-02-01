function Reward=RewardCalculate(RewardMatrix,Customer2Item)
%Calculate the reward of a Customer Assignment  
mysum=0;%Set reward to zero
NofCustomers=length(Customer2Item);
for i=1:NofCustomers%For all customers
    mysum=mysum+RewardMatrix(i,Customer2Item(i));%sum the reward of thwe item assigned to the customer
end
Reward=mysum;%return the sum