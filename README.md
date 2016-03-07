# modify-model
Note that the following code is not meant for public use. That is, you can use it, but I will provide little documentation.

# Usage
This tool opens a caffemodel (V2) file, and changes the shape of specified layers. 
Missing data maybe repeated or averaged from existing channels, extraneous data will be deleted.

Usage example for modification:

modify --input input.caffemodel --output output.caffemodel

Usage example for simply printing layers:

modify --input input.caffemodel --view

No guarantees are given ;)
