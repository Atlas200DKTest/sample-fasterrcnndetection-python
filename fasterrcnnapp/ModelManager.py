# encoding=utf-8
import hiai


class ModelManager(object):
    def __init__(self):
        pass

    def CreateGraph(self, model, model_width, model_height, dvpp_width, dvpp_height):
        myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
        if myGraph is None:
            print
            'get defaule graph failed'
            return None
        cropConfig = hiai.CropConfig(0, 0, dvpp_width, dvpp_height)
        print 'cropConfig ', cropConfig
        resizeConfig = hiai.ResizeConfig(model_width, model_height)
        print 'resizeConfig ', resizeConfig
        nntensorList = hiai.NNTensorList()
        print 'nntensorList', nntensorList
        resultCrop = hiai.crop(nntensorList, cropConfig)
        print 'resultCrop', resultCrop
        resultResize = hiai.resize(resultCrop, resizeConfig)
        print 'resultResize', resultResize
        resultInference = hiai.inference(resultResize, model, None)
        print 'resultInference', resultInference
        if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
            print 'create graph ok !!!!'
            return myGraph
        else:
            print 'create graph failed, please check Davinc log.'
            return None

    def CreateGraphWithoutDVPP(self, model):
        myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
        if myGraph is None:
            print 'get defaule graph failed'
            return None

        nntensorList = hiai.NNTensorList()
        if nntensorList is None:
            print('nntensor is None')
            return None

        resultInference = hiai.inference(nntensorList, model, None)
        if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
            print 'create graph ok !!!!'
            return myGraph
        else:
            print 'create graph failed!'
            return None

    def Inference(self, graphHandle, inputTensorList):
        if not isinstance(graphHandle, hiai.Graph):
            print "graphHandle is not Graph object"
            return None
        resultList = graphHandle.proc(inputTensorList)
        return resultList
