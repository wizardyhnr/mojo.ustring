# mojo.ustring

Practice code for Mojo.

Mojo's String encoding is utf-8, which has varying code length for char. Slicing operation of String is on byte order, which is not "legal" char index.
Slicing of String will need to decode String from beginning. This makes random access through indexing not as convinient as Python.

In ustring.mojo, UString is a struct wrapped around String with additional fields: length (length of chars in String), inner_string (utf-8 encoded String), and idx (List[UInt8], which records corresponding byte index (end position) of each char in segments, each segment's lenght is 64 to ensure byte index < 255).

When length of UString is larger than 64, slicing calculation involes accumulation of previous indexs.

ustring.mojo implements __getitem__() to allow slicing based on char index.

test_UString.mojo contains test cases for ustring.mojo

char length calculation borrows ideas from https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d

byte index calc for each char is based on prefix sum computation on SIMD. 

slicing ideas comes from source code of string.mojo in stdlib of Mojo.

Code heavily leverages on SIMD operation offered in Mojo to speed up. 

Additional storage cost for each char is one byte. 

Initially, I was thinking about only recoding length of each char to save storge. lenght of possible utf-8 code is 1, 2, 3, 4. So if __mlir_type.i2 like __mlir_type.i1 is implementalbe, then storage cost could be reduced to 1/4 byte per char. I don't understand inner implementation of __mlir_type.i1. current ustring.mojo uses List[UInt8] to record byte index instead. 

Todo:

implement inplace dunder method like __iadd__() in String since modified inner_string needs idx and length update.
