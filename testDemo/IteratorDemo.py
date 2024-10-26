my_list=[1,2,3,4,5]
my_iter=iter(my_list)
print(next(my_iter))

class MyIterator:
    def __init__(self,start,end):
        self.current=start
        self.end=end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current<=self.end:
            current=self.current
            self.current+=1
            return current
        else:
            raise StopIteration

my_iter_new=MyIterator(1,5)
for num in my_iter_new:
    print(num)
