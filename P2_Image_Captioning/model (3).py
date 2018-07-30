import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        #define parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # word embedding
        self.embedding = nn.Embedding(vocab_size,embed_size)
        # LSTM
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        # FC
        self.linear = nn.Linear(self.hidden_size,vocab_size) 

    
    def forward(self, features, captions):
        embeded = self.embedding(captions[:,:-1]) 
        input_combined = torch.cat((torch.unsqueeze(features,1),embeded),1)        
        output,hidden = self.lstm(input_combined)        
        output = self.linear(output)        
      
        return output

    def sample(self, inputs,beam_size=20, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #try use beam search with beam size = 10 ，feed an image and output a best senteces from k sets.
        # There are 3 main variables need to take care:log probability, word_idx, hidden state
        start_word = 0       
        end_word = 1
        # feed image into trained decoder.inputs.shape=[1,1,512]
        y0, hidden = self.lstm(inputs)                
        #y_next=F.log_softmax(torch.squeeze(self.linear(y0)),dim=0)
        #new_state = hidden

        if beam_size > 1:

            # get the first word by feed the hidden and start_word from image output.  y_next.shape = [9955]
            y_next, new_state = self.lstm_cell(start_word,hidden)
            
            # Get the top words with beam size and store in beams.
            top_prob,top_idx = torch.sort(y_next, descending=True)             
            beams = []
            completed_sentence = []  
            for i in range(beam_size):                
                    beams.append((top_prob[i].item(), [top_idx[i].item()],  new_state))

            # run beam search
            nstep = 0
            while True:
                    candidates = []
                    for b in beams:                        
                        # beams=[(log_prob_sum, [word_idx], pre_hidden)]                             
                        pre_word = b[1][-1]  
                        
                        # the length of sentence generated so far
                        lenth = len(b[1]) 
                        
                        # check if sentence completed and remove it from beams.
                        if  lenth == max_len or pre_word == end_word: 
                            completed_sentence.append(b[:-1])                           
                            del b 
                            
                        # keep searching if not meet the condition    
                        elif lenth < max_len or pre_word != end_word:     
                            
                           # feed prev word and states, get next word and stats
                            y_next,new_state = self.lstm_cell(pre_word,b[2])
                            top_prob, top_idx = torch.sort(y_next, descending=True)
                            # calculate all probability for next word. Or for save time reduce to beam size?
                            for i in range(beam_size):
                                #sum(log prob) * 1/Ty to reduce length penalty according to NG's course
                                log_prob = (b[0] + top_prob[i].item())/lenth
                                word_idx = b[1] + [top_idx[i].item()]
                                candidates.append((log_prob,word_idx,new_state))                                    
        
                    # Chose the top score words and store in beams
                    candidates.sort(reverse=True)                    
                    beams = candidates[:beam_size]
                    nstep += 1
                    # Stop searching once meet below condition.
                    if  len(completed_sentence)>= beam_size or nstep == max_len:
                        break
            
            # Order by log prob
            completed_sentence.sort(reverse=True)
            
        else:
            # Greedy search
            sentence = []
            # Get the first word
            prev_word = F.log_softmax(torch.squeeze(self.linear(y0)),dim=0).max(0)[1].item()
            for i in range(max_len):
                # Get the next word
                next_word, hidden = self.lstm_cell(prev_word,hidden)
                # Find the max prob word index
                next_word = next_word.max(0)[1].item()
                # Add in sentence list.
                sentence.append(next_word)
                prev_word = next_word
            return sentence             
                
                                               
        return completed_sentence[0][1]
           
     
    def lstm_cell(self,word,hidden):
        # generate next word according to gived word and hidden
        word = torch.cuda.LongTensor([word])
        embeded = torch.unsqueeze(self.embedding(word),1)
        output, h = self.lstm(embeded,hidden)
        output = torch.squeeze(self.linear(output))    
        return F.log_softmax(output,dim=0),h
        
            

        
        
            
        
        
        
        