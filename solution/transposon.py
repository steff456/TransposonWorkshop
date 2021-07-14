# Clase de transposon
class Transposon():
    # Definir inicializacion del objeto
    def __init__(self, seq_name, first, last, score):
        self.sequence_name = seq_name
        self.first = first
        self.last = last
        self.score = score

    # Definir si hay sobrelape con otro transposon
    def is_overlap(self, transposon):
        if self.first <= transposon.last <= self.last:
            return True
        elif self.first <= transposon.first <= self.last:
            return True
        else:
            return False

    # Definir el tamaño del sobrelape
    def get_overlap(self, transposon):
        return max(0, min(self.last-transposon.first,
                          transposon.last-self.first,
                          len(self), len(transposon)))

    # Retornar la longitud del transposon
    def __len__(self):
        return self.last - self.first + 1

    # Retornar la comparacion de la fiabilidad con otro transposon
    def __gt__(self, transposon):
        return self.score > transposon.score

    # Retornar la definicion de igualdad con otro transposon
    def __eq__(self, transposon):
        return self.score == transposon.score

    # Definir la representacion del String
    def __str__(self):
        return '{}\t{}\t{}'.format(self.sequence_name, self.first, self.last)
