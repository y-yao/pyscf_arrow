! This checks to see if FCIDUMPs generated by pyscf+Dice has the correct symmetry labels (ones that satisfy angular momentum conservation).
! It should be run only on FCIDUMPs that have not been compactified to save only 1/8th of the integrals
! (one can save only 1/8th if one does explicit angular momentum checking which sqmc does but Dice does not).
! If the integrals have been compactified, then this routine will say they violate angular momentum conservation, even though it will work with sqmc.
! It does upto 8 iterations and 3 techniques (steps).
! step 1:  It looks at 1-body integrals and for each abs(orbital_symmetry) it flips the sign of the orbital_symmetry for those orbs with
!          largest number of violations.  This can fail (oscillated back and forth), if half the orbs of a given orbital_symmetry are wrong.
!          If num_switch=0 on any iteration, it jumps immediately to step 3.  Otherwise it goes to step 2 after 4 iterations.
! step 2:  It looks at 1-body integrals and flips the signs of the violator and its partner, as soon as the violation is detected.
!          If num_switch=0, it goes to step 3.  At this point orbs of each abs(orbital_symmetry) are consistent but there may be inconsistencies
!          between different symmetries.
!          If num_switch.ne.0 after 6 iterations (2 of them at step 2) then it seems likely that pyscf broke symmetry, so give up.
! step 3:  It looks at 2-body integrals and when it detect a violation it switches the sign of all the orbitals of the abs(orbital_symmetry)
!          of the highest index orbitals of the 4 orbitals in the 2-body integral.
! The bottom line is that this code always works except when pyscf breaks symmetry in which case it is impossible to fix the violations.

implicit none
integer orbital_symmetry(0:2000), lz(0:2000), nviolated(0:2000), norb, ind(4), loc_max, ind_max, num_1body, num_switch, num_switch_tot, orbital_symmetry_max, max_iter, iter, step, i, j, n
integer, allocatable :: max_nviolated_orbital_symmetry(:)
character*2048 string
character*11 string1
character*7 string2
real*8 integral
real*8, parameter :: eps=1e-6

step=1
num_switch_tot=0
max_iter=8
do iter=1,max_iter
  write(6,'(/,''fix orb_symmetries iteration'',i3,'' step '',i3)') iter, step
  if(step==1) write(6,'(''Checking 1-body integrals and flipping the orbs with most violations for each abs(orbital_symmetry)'',/)')
  if(step==2) write(6,'(''Checking 1-body integrals and flipping them as we find violations'',/)')
  if(step==3) write(6,'(''Checking 2-body integrals and flipping them as we find violations'',/)')
  if(iter==1 .or. step==3) then
    rewind 5
    
    read(5,'(a11)',advance='no') string1
    if(iter==1) write(6,*) string1
    if(string1.ne.'&FCI NORBS=') stop '1st line of FCIDUMP does not begin with &FCI NORBS='
    read(5,*) norb
    if(iter==1) write(6,'(''norb='',i5)') norb
    read(5,'(a7)',advance='no') string2
!   if(iter==1) write(6,*) string2
    if(string2.ne.'ORBSYM=') stop '2nd line of FCIDUMP does not begin with ORBSYM='
    if(iter==1) then
      read(5,*) orbital_symmetry(1:norb)
    endif
    
    do i=1,norb
      if(abs(orbital_symmetry(i)).le.2) then
        lz(i)=0
      else
        lz(i)=sign(((abs(orbital_symmetry(i))-5)/2+1),orbital_symmetry(i))
      endif
    enddo
    if(iter==1) write(6,'(''ind'',1000i3)') (i,i=1,norb)
    if(iter==1) write(6,'(''sym'',1000i3)') orbital_symmetry(1:norb)
    if(iter==1) write(6,'(''Lz '',1000i3)') (lz(i),i=1,norb)
    
    do ! Read rest of header part of FCIDUMP
      read(5,'(A)') string
      if(len_trim(string)==len(string)) write(6,'(''Warning: string not long enough'')')
      if(iter==1) write(6,*) trim(string) ; call flush(6)
      if(index(string,'&').ne.0 .or. index(string,'/').ne.0) exit
    enddo
  else ! if we are step 1 or 2, then instead of rewinding just backspace over the 1-body integrals
    write(6,'(''num_1body back stepped='',i8)') num_1body
    do i=1,num_1body+1
      backspace 5
    enddo
  endif
  
  num_switch=0
  nviolated(1:norb)=0
  num_1body=0
  do ! Read the integral values and indices of orbs until end of FCIDUMP
    read(5,*,end=98) integral, ind(1:4)
    if(ind(3).eq.0 .and. ind(4).eq.0) then !   1-body integrals
      num_1body=num_1body+1
      if(orbital_symmetry(ind(1)).ne.orbital_symmetry(ind(2)) .and. abs(integral).gt.eps) then
!       write(6,'(''angular momentum not conserved for'',4i4,'' lz='',2i4,'' integral='',f16.12)') ind, (lz(ind(i)),i=1,2), integral
        do i=1,2
          nviolated(ind(i))=nviolated(ind(i))+1
        enddo
        if(step==2) then
          loc_max=maxloc(abs(ind(1:2)),1)
          ind_max=ind(loc_max)
          ! Figure out whether to switch the orbital_symmetry of ind_max with ind_max-1 or ind_max+1
          do n=1,ind_max-1
            if(abs(orbital_symmetry(ind_max-n)) .ne. abs(orbital_symmetry(ind_max))) then
              exit
            endif
          enddo
          !write(6,'(''n,sym='',9i5)') n, orbital_symmetry(ind_max-1), orbital_symmetry(ind_max), orbital_symmetry(ind_max+1)
          if(mod(n,2)==0) then
            n=-1
          else
            n=1
          endif
          ! Do the switch
          num_switch=num_switch+1
          if(orbital_symmetry(ind_max)==-orbital_symmetry(ind_max+n)) then
            write(6,'(''switching sym of orbitals='',2i4,'' with sym'',9i4)') ind_max,ind_max+n, orbital_symmetry(ind_max), orbital_symmetry(ind_max+n)
            orbital_symmetry(ind_max)=orbital_symmetry(ind_max+n)
            orbital_symmetry(ind_max+n)=-orbital_symmetry(ind_max)
            lz(ind_max)=lz(ind_max+n)
            lz(ind_max+n)=-lz(ind_max)
          else
            write(6,'(''ind_max, orbital_symmetry(ind_max-1), orbital_symmetry(ind_max), orbital_symmetry(ind_max+1), integral'',i5,3i4,f16.8)') ind_max, orbital_symmetry(ind_max-1), orbital_symmetry(ind_max), orbital_symmetry(ind_max+1), integral
            write(6,'(''In 1-body integs, orbital_symmetry(ind_max)!=-orbital_symmetry(ind_max+n)'')')
            exit
          endif
        endif
      endif
    else
!     2-body integrals
      if(step==3) then
        if(lz(ind(1))+lz(ind(3)).ne.lz(ind(2))+lz(ind(4)) .and. abs(integral).gt.eps) then
          write(6,'(''angular momentum not conserved for'',4i4,'' orbital_symmetry='',4i4,'' lz='',4i4,'' integral='',f16.12)') ind, (orbital_symmetry(ind(i)),i=1,4), (lz(ind(i)),i=1,4), integral
          loc_max=maxloc(abs(ind),1)
          ind_max=ind(loc_max)
          do i=1,norb
            if(abs(orbital_symmetry(i))==abs(orbital_symmetry(ind_max))) then
!             write(6,'(''switching sym of orbital='',i4,'' with sym'',9i4)') i, orbital_symmetry(i)
              num_switch=num_switch+1
              orbital_symmetry(i)=-orbital_symmetry(i)
              lz(i)=-lz(i)
            endif
          enddo
        endif ! lz(ind(1))+lz(ind(3)).ne.lz(ind(2))+lz(ind(4))
      endif ! step=3
    endif ! end 2-body integrals
  enddo ! read integrals

  98 continue 
  if(step==2) write(6,'(/,''num_switch2='',i5)') num_switch
  if(step==3) write(6,'(/,''num_switch3='',i9)') num_switch
  if(num_switch.ne.0 .and. step==2) then
    write(6,'(''ind'',1000i6)') (i, i=1,norb)
    write(6,'(''vio'',1000i6)') nviolated(1:norb)
    write(6,'(''sym'',1000i6)') (orbital_symmetry(i),i=1,norb)
    write(6,'(''lz '',1000i6)') (lz(i),i=1,norb)
  endif
  if(step==3 .and. num_switch==0) exit
  if(step==2 .and. num_switch==0) then
    write(6,'(''1-electron integrals can be made to conserve angular momentum'')')
    step=3
  endif 
  if(iter==6 .and. num_switch.ne.0) then
    write(6,'(''Cannot make the symmetry labels satisfy symmetry.  Pyscf probably broke symmetry. Check if there is a better starting state.'')')
    exit
  endif 
  
  ! Here we look to see for each abs(orbital_symmetry) which orbs have the most violations or 1-body integrals and we flip those.
  if(step==1) then
    orbital_symmetry_max=maxval(abs(orbital_symmetry(1:norb)),1)
    if(.not.(allocated(max_nviolated_orbital_symmetry))) allocate(max_nviolated_orbital_symmetry(orbital_symmetry_max))
    max_nviolated_orbital_symmetry(1:orbital_symmetry_max)=0
 
    ! Find the maximum number of violations for each abs(orbital_symmetry)
    do i=1,orbital_symmetry_max
      do j=1,norb
        if(abs(orbital_symmetry(j))==i .and. nviolated(j).gt.max_nviolated_orbital_symmetry(i)) then
          max_nviolated_orbital_symmetry(i)=nviolated(j)
        endif
      enddo
    enddo
    write(6,'(''max_nviolated_orbital_symmetry='',100i5)') max_nviolated_orbital_symmetry(1:orbital_symmetry_max)
    
    ! Switch orbital symmetries for those orbs that have the maximum number of violations for each abs(orbital_symmetry)
    num_switch=0
    do i=1,orbital_symmetry_max
      do j=1,norb,2
        if(abs(orbital_symmetry(j))==i .and. nviolated(j).eq.max_nviolated_orbital_symmetry(i) .and. max_nviolated_orbital_symmetry(i).ge.1) then
          ! Figure out whether to switch the orbital_symmetry of j with j-1 or j+1
          do n=1,j-1
            if(abs(orbital_symmetry(j-n)) .ne. i) then
              exit
            endif
          enddo
          if(mod(n,2)==0) then
            n=-1
          else
            n=1
          endif
          ! Do the switch
          num_switch=num_switch+1
          if(orbital_symmetry(j)==-orbital_symmetry(j+n)) then
            write(6,'(''for orbital_symmetry='',i3,'' switching sym of orbitals='',2i4,'' with sym'',9i4)') i,j,j+n, orbital_symmetry(j), orbital_symmetry(j+n)
            orbital_symmetry(j)=orbital_symmetry(j+n)
            orbital_symmetry(j+n)=-orbital_symmetry(j)
            lz(j)=lz(j+n)
            lz(j+n)=-lz(j)
          else
            write(6,'(''j, orbital_symmetry(j-1), orbital_symmetry(j), orbital_symmetry(j+1)'',i5,3i4)') j, orbital_symmetry(j-1), orbital_symmetry(j), orbital_symmetry(j+1)
            write(6,'(''In 1-body integs, orbital_symmetry(ind_max)!=-orbital_symmetry(ind_max+n)'')')
            write(6,'(''orbital_symmetry(j)!=-orbital_symmetry(j+1) .and. orbital_symmetry(j)!=-orbital_symmetry(j-1)'')')
            exit
          endif
        endif
      enddo
    enddo
    if(num_switch==0) then
      write(6,'(''1-electron integrals can be made to conserve angular momentum'')')
      step=3
    elseif(iter==4) then
      step=2
    endif
    write(6,'(/,''num_switch1='',i5)') num_switch
  endif ! step=1
  
  num_switch_tot=num_switch_tot+num_switch
  
  if(iter==max_iter.and.num_switch.ne.0) then
    write(6,'(''Warning: iter=max_iter.and.num_switch.ne.0'')')
  endif

enddo ! iter

write(6,*)
if(num_switch_tot==0) then
  write(6,'(''original FCIDUMP conserved angular momentum'')')
else
  write(6,'(''original FCIDUMP did NOT conserve angular momentum'')')
endif
if(num_switch==0) then
  write(6,'(''final FCIDUMP conserved angular momentum'')')
  write(6,'(''ORBSYM=    '',1000i4)') orbital_symmetry(1:norb)
else
  write(6,'(''final FCIDUMP MAY NOT conserve angular momentum'')')
endif
write(6,'(''-------------------------------------------------'')')

stop
end
