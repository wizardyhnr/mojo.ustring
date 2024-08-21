from ustring import UString
from sys.info import simdbitwidth

fn main():
    var test_string = "g测试开始需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串需要一个足够长的字符串"
    var myString = UString(test_string)
    print(test_string)
    print("char length is:", len(myString))
    print("byte length is:", len(test_string))
    myString._update_idx()
    print(str("===")*6)
    
#var str2 = "idx:\n"
#    for i in range(len(myString)):
#        print(myString.idx[i]) 
    var i = -1
    print("byte idx of ", i, "th char is", myString.inner_string[myString._calc_slice(i)], myString._calc_slice(i))
    i = 0
    print("byte idx of ", i, "th char is", myString.inner_string[myString._calc_slice(i)], myString._calc_slice(i))
    i = 12
    print("byte idx of ", i, "th char is", myString.inner_string[myString._calc_slice(i)], myString._calc_slice(i))
    print(myString[0:5:2])
    print(myString[0:5])
    print(myString[2:2])
    print("average char length is:", len(test_string)/len(myString))
    print("bit width of SIMD on current platform is:", simdbitwidth())
