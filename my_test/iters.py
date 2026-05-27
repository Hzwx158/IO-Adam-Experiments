from typing import Generator, Iterator


class permutation:
    def __init__(self, *iters, iter_transform=None):
        '''
        Permulate some iterable objects. Use this to avoid ugly multiple `for`
        Params:
            *iters: iterable objects
            iter_transform: function to pack each iterator. etc. enumerate
        Example:
        >>> a=[1,2,3], b='abcd'
        >>> for (i,j) in permutation(a,b):
        ...     pass
        
        Same as:
        >>> for i in a:
        ...     for j in b:
        '''
        # print(iters)
        self.iters = iters
        self.make_iter = iter if iter_transform is None else lambda i:iter(iter_transform(i))
    def __iter__(self):
        self.__activate_iters = [self.make_iter(i) for i in self.iters]
        self.__last_res = [None for _ in self.__activate_iters]
        return self
    def __next__(self)->tuple:
        if self.__activate_iters is None or len(self.__activate_iters)==0:
            raise StopIteration
        if self.__last_res[0]==None:
            self.__last_res = [next(it) for it in self.__activate_iters]
            return tuple(self.__last_res)
        res = self.__last_res[0:-1]
        res.append(None)
        for idx in range(len(self.__activate_iters)-1, -1, -1):
            try:
                res[idx] = next(self.__activate_iters[idx])
                self.__last_res = res
                return tuple(res)
            except StopIteration:
                if idx==0: #全部遍历完了，结束
                    break
                # [idx] 遍历完了, [idx-1]应该递进一个
                self.__activate_iters[idx] = self.make_iter(self.iters[idx])
                res[idx] = next(self.__activate_iters[idx])
                
        self.__activate_iters = self.__last_res = None
        raise StopIteration
    def __len__(self):
        res = 1
        for i in self.iters:
            res *= len(i if hasattr(i, '__len__') else list(i))
        return res

class n_range:
    def __init__(self, *dims):
        '''
        Permulate some range(...). Use this to avoid ugly multiple `for`
        Params:
            *dims(tuple[int] or Iterable): length of each dim.
        Returns:
            A generator, will return a list[int]
        Example:
        >>> for (i1,i2,i3) in n_range(2,3,4):
        ...     print(i1,i2,i3)
        
        Same as:
        >>> for i1 in range(dim1):
        ...  for i2 in range(dim2):
        ...   for i3 in range(dim3):
        '''
        self.dims = dims
    def __iter__(self):
        return iter(permutation(*[range(d) for d in self.dims]))


if __name__=='__main__':
    abc = permutation([1,2],'ab',iter((4,5,6)))
    print(len(abc))