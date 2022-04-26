# Functions to freeze certain layers in the input model
import math

def percentage_to_freeze(model):
    # Freezes model paramaters (top down) for the given percentage
    percentage_to_freeze = int(input('What percent of the model should be frozen?: '))
    freeze_number = math.ceil((1-(1/100*percentage_to_freeze))*len(list(model.named_parameters())))
    print('Number of paramaters frozen: ',len(list(model.named_parameters()))-freeze_number)
    print('The following parameters remain frozen:')
    for (name, param) in list(model.named_parameters())[:-freeze_number]:
        print(name)
        param.requires_grad = False

def freeze_by_children(model):
    # Freezes model elements 
    print('The input model has',len(list(model.named_children())),'children')
    for idx,(item,module) in enumerate(list(model.named_children())):
        print(idx+1,'-',item)
    number_to_freeze = int(input('How many layers would you like to freeze (top down): '))
    if number_to_freeze<= len(list(model.named_children())):
        
        count = 1
        for child in model.children():
            if count<=number_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
                print(child,'has been unfrozen')
            count+=1
    else:
        print('Invalid Number')